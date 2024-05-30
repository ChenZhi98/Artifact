import csv
import json
import multiprocessing
import os
import random
import re
import time
from collections import defaultdict

import evaluate
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from arguments import GenerationArguments

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path = "../models/trained/federated/round_10"
# export trust_remote_code=True

# accelerate launch --gpu_ids='all'  generation.py --do_sample True --temperature 0.6 --top_p 0.6 --n_samples=5  --model_ckpt="../models/trained/" --prompts_file="prompts/google_functions_info.csv" --generations_save_file="generated/fl_gg_generated_content.csv" --HF_ALLOW_CODE_EVAL="1"

class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            prompts.append(self.tokenizer.eos_token + self.dataset[task]["signature with docstring"].strip())
        outputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "function_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }

def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def postprocess_generation(prompt, generation, stop_words):
    generation = generation[len(prompt) :]
    return stop_at_stop_token(generation, stop_words)

def complete_code(accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, **gen_kwargs):
    prompt_token_dict =  defaultdict(list)
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"][:, : batch["input_len"]], num_return_sequences=batch_size, **gen_kwargs
            )


            generated_tasks = batch["function_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather((generated_tokens, generated_tasks))
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                ids=batch["ids"][0].tolist()

                gen_token_dict[task].append(generated_tokens)
                prompt_token_dict[task].append(ids)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            prompt = tokenizer.decode(prompt_token_dict[task][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_code = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_code = postprocess_generation(prompt,gen_code,EOF_STRINGS)

            code_gens[task].append(gen_code)

    return code_gens


def main():
    # Setup configuration
    parser = HfArgumentParser(GenerationArguments)
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = args.HF_ALLOW_CODE_EVAL
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    # Use dataset load to feed to accelerate
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)

    model.to(device)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }

    # Load evaluation dataset and metric
    functions_info = load_dataset('csv', data_files={'test': args.prompts_file})

    n_tasks = args.num_tasks if args.num_tasks is not None else len(functions_info["test"])
    n_copies = args.n_samples // args.batch_size

    functions_info_tokenized = TokenizedDataset(tokenizer, functions_info["test"], n_copies=n_copies, n_tasks=n_tasks)
    functions_info_loader = DataLoader(functions_info_tokenized, batch_size=1)

    model, functions_info_loader = accelerator.prepare(model, functions_info_loader)

    all_generations = complete_code(
        accelerator,
        model,
        tokenizer,
        functions_info_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        **gen_kwargs,
    )

    print("==========================================================================")
    # Open a CSV file for writing
    with open(args.generations_save_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['function id', 'generation id', 'generated content'])
        for function_index, function_generations in enumerate(all_generations, start=1):
            for function_generation_index, function_generation in enumerate(function_generations, start=1):
                writer.writerow([function_index, function_generation_index, function_generation])

if __name__ == "__main__":
    main()