import argparse
import logging
import os
import time
import warnings
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
import datasets
import flwr as fl
import torch
import os
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed
from arguments import TrainingArguments

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# accelerate launch --gpu_ids='all' client.py --node_id=1 --local_dataset_path="../../dataset/microsoft" --dataset_name_train="microsoft_train.json.gz" --dataset_name_valid="microsoft_valid.json.gz"
# accelerate launch --gpu_ids='all' client.py --node_id=2 --local_dataset_path="../../dataset/google" --dataset_name_train="google_train.json.gz" --dataset_name_valid="google_valid.json.gz"
# accelerate launch --gpu_ids='all' client.py --node_id=3 --local_dataset_path="../../dataset/facebook" --dataset_name_train="facebook_train.json.gz" --dataset_name_valid="facebook_valid.json.gz"

# Settings
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

# Accelerator
project_config = ProjectConfiguration(project_dir=args.save_dir, logging_dir="log")
accelerator = Accelerator(log_with=["wandb"], project_config=project_config)
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

args = Namespace(**vars(args), **acc_state)
set_seed(args.seed)


def setup_logging(args):
    project_name = args.project_name
    logger = logging.getLogger(__name__)
    log_dir = Path(args.save_dir) / "log/"
    log_dir.mkdir(exist_ok=True)
    filename = f"debug_{accelerator.process_index}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / filename), logging.StreamHandler()],
    )
    if accelerator.is_main_process:  
        accelerator.init_trackers(project_name, vars(args))
        run_name = accelerator.trackers[0].run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name

# Logging
logger, run_name = setup_logging(args)
logger.info(accelerator.state)

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
            self,
            tokenizer,
            dataset,
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            tokenized=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=True)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)

def get_grouped_params(model, args, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def log_metrics(metrics):
    logger.info(metrics)
    if accelerator.is_main_process:
        accelerator.log(metrics)

def compute_tflops(elapsed_time, accelerator, args, model, tokenizer):
    config_model = accelerator.unwrap_model(model).config
    checkpoint_factor = 4 if args.gradient_checkpointing else 3
    batch_size = args.train_batch_size * accelerator.state.num_processes * args.gradient_accumulation_steps
    factor = 24 * checkpoint_factor * batch_size * args.seq_length * config_model.n_layer * (config_model.n_embd ** 2)
    flops_per_iteration = factor * (
            1.0
            + (args.seq_length / (6.0 * config_model.n_embd))
            + (tokenizer.vocab_size / (16.0 * config_model.n_layer * config_model.n_embd))
    )
    tflops = flops_per_iteration / (elapsed_time * accelerator.state.num_processes * (10 ** 12))
    return tflops


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def create_dataloaders(args, tokenizer):
    ds_kwargs = {"streaming": True}

    train_data = load_dataset(path=args.local_dataset_path, data_files=f'{args.dataset_name_train}', split="train",
                              **ds_kwargs)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    valid_data = load_dataset(path=args.local_dataset_path, data_files=f'{args.dataset_name_valid}', split="train",
                              **ds_kwargs)


    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    )


    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    return train_dataloader, eval_dataloader


def train(node_id, current_round ,model, tokenizer,train_dataloader, optimizer, lr_scheduler):
    model.train()
    completed_steps = 0
    t_start = time.time()
    loss_tracking = 0

    max_epochs = args.max_train_epochs
    step = 0
    current_epoch = 0
    trained_samples = 0

    while current_epoch <= max_epochs:
        current_epoch += 1

        for batch in train_dataloader:
            step += 1
            loss = model(batch, labels=batch, use_cache=False).loss
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            loss_tracking += avg_loss.item() / args.gradient_accumulation_steps

            trained_samples = step * accelerator.state.num_processes * args.train_batch_size

            log_metrics({"node id":node_id, "current round":current_round, "local step": step, "local epoch": current_epoch ,"samples": trained_samples, "loss_per_step/train": loss.item()})
            loss = loss / args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps != 0:
                # Prevent backward from doing gradient all_reduce in every step
                if accelerator.distributed_type == DistributedType.MULTI_GPU:
                    with model.no_sync():
                        accelerator.backward(loss)
                else:
                    accelerator.backward(loss)
            else:
                lr = get_lr(optimizer)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                elapsed_time = time.time() - t_start
                tflops = compute_tflops(elapsed_time, accelerator,  args, model, tokenizer)
                completed_steps += 1
                log_metrics(
                    {   "node id":node_id, "current round":current_round, "local step": step, "local epoch": current_epoch ,
                        "completed_steps": completed_steps,
                        "loss/train": loss_tracking,
                        "lr": lr,
                        "tflops": tflops,
                        "time_per_iteration": elapsed_time,
                    },
                )
                t_start = time.time()
                loss_tracking = 0

    return trained_samples


def evaluate(args, model, eval_dataloader):
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    model.eval()
    losses = []
    total_steps = 0

    for step, batch in enumerate(eval_dataloader):
        total_steps = step

        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if 0 < args.max_eval_steps <= step:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    evaluated_samples = total_steps*args.valid_batch_size
    
    print(f"evaluated_samples, {evaluated_samples}. loss.item(){loss.item()}, perplexity.item():{perplexity.item()}")

    return evaluated_samples, loss.item(), perplexity.item()


def main():
    node_id = args.node_id
    logger.info("node_id: %s", node_id)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.base_model_dir)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Load dataset and dataloader
    train_dataloader, eval_dataloader = create_dataloaders(args, tokenizer)

    # Prepare the optimizer and learning rate scheduler
    optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Flower client
    class Client(fl.client.NumPyClient):

        def get_parameters(self, config):
            logger.info("Return Parameters to Aggregator.")
            logger.info("config: %s", config)
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            logger.info("Set Parameters to Client Model.")
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            # Read values from config
            server_round = config["server_round"]

            # Use values provided by the config
            logger.info("[round %s] fit, config: %s", server_round, config)

            self.set_parameters(parameters)
            logger.info("Training Started.")
            trained_samples = train(node_id, server_round, model, tokenizer, train_dataloader, optimizer, lr_scheduler)
            logger.info("Training Finished.")

            return self.get_parameters(config={}), trained_samples, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            evaluated_rows, loss, perplexity = evaluate(args, model, eval_dataloader)

            server_round = config["server_round"]

            if isinstance(server_round, str) and server_round.isdigit():
                server_round = int(server_round)

            # Save model for each round
            if node_id == args.save_model_node_id:
                logger.info(f"Save model for node {node_id} in round {server_round}")
                accelerator.wait_for_everyone()
                save_dir = os.path.join(args.save_dir, f"round_{server_round}")
                unwrapped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
                unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)

            return float(loss), evaluated_rows, {"perplexity": float(perplexity)}

    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8888",
        grpc_max_message_length=-1,
        client=Client().to_client(),
    )


if __name__ == "__main__":
    main()
