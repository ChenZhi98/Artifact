from accelerate import Accelerator, DistributedType
from datasets import load_dataset, IterableDataset
from torch import device
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
import torch
from arguments import PerplexityArguments
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader, logger
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
from datasets import concatenate_datasets
import logging
import os



# Setup configuration
parser = HfArgumentParser(PerplexityArguments)
args = parser.parse_args()

# accelerate launch --gpu_ids='all' evaluate_perplexity.py --model_dir="../models/trained/incremental_m2g"
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
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

    def __len__(self):
        return len(self.dataset)
        
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
                        logger.info("Epoch Finished")
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

def create_dataloaders():
    ds_kwargs = {}

    data_files = ["google_valid.json.gz","microsoft_valid.json.gz","facebook_valid.json.gz"]

    datasets = []
    for data_file in data_files:
        dataset = load_dataset(path=args.local_dataset_path, data_files=f'{data_file}', split="train", **ds_kwargs)
        datasets.append(dataset)

    valid_data = concatenate_datasets(datasets)
    valid_data = valid_data.shuffle(seed=42)
    
    print(f"Total size of the dataset: {valid_data.num_rows}")

    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    )


    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    return eval_dataloader


def evaluate_model(model, eval_dataloader):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluating"):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


model = AutoModelForCausalLM.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)


eval_dataloader = create_dataloaders()

config = ProjectConfiguration(project_dir=args.save_dir, logging_dir="log")
accelerator = Accelerator(log_with=["wandb", "tensorboard"], project_config=config)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(
    model, eval_dataloader
)

# Create the output directory if it does not exist
output_dir = "perplexity_results"
os.makedirs(output_dir, exist_ok=True)

# Create the output file path
output_file_path = os.path.join(output_dir, "perplexity_log")

# Set up logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[logging.FileHandler(output_file_path, mode='a'), logging.StreamHandler()])

# Replace print statements with logging.info
logging.info(f"evaluate perplexity for model: {args.model_dir}")
loss, perplexity = evaluate_model(model, eval_dataloader)
logging.info(f"Loss: {loss:.4f}, Perplexity: {perplexity:.2f}")
