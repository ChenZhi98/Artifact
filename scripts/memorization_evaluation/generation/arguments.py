from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationArguments:
    """
    Configuration for generating codes.
    """
    model_ckpt: Optional[str] = field(
        default="../models/trained/", metadata={"help": "Model name or path of model to be evaluated."}
    )
    tokenizer_dir: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Path of tokenizer."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    num_tasks: Optional[int] = field(
        default=None,
        metadata={"help": "The number of tasks to run. If not included all tasks are evaluated."},
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Sample from the language model's output distribution."}
    )

    temperature: Optional[float] = field(default=0.6, metadata={"help": "Sampling temperature used for generation."})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "Maximum number of newly generated tokens."})
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.6, metadata={"help": "Top-p parameter used for nucleus sampling."})
    repetition_penalty: Optional[float] = field(default=1.1, metadata={"help": "Repetition_penalty parameter helps to discourage the model from repeating the same token by decreasing the probability of tokens that have already appeared."})

    batch_size: Optional[int] = field(default=5, metadata={"help": "Number of generations to run in parallel."})
    n_samples: Optional[int] = field(
        default=5, metadata={"help": "Number of completions to generate for each sample."}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})
    HF_ALLOW_CODE_EVAL: Optional[str] = field(
        default="0", metadata={"help": "Allow `code_eval` to execute Python code on machine"}
    )
    device_int: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Determine which device to run the `text-generation` Pipeline on. -1 is CPU and any zero or positive"
                " number corresponds to which GPU device id to run on."
            )
        },
    )
    prompts_file: Optional[str] = field(
        default="prompts/_.csv", metadata={"help": "Prompts file name"}
    )
    generations_save_file: Optional[str] = field(
        default="generated/generated.csv", metadata={"help": "File name for saving the generated contents"}
    )


