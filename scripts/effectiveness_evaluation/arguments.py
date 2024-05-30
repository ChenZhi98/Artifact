from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PerplexityArguments:
    model_dir: Optional[str] = field(
        default="../models/trained/microsoft/epoch_10", metadata={"help": "Model name or path of model"}
    )
    tokenizer_dir: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Path of tokenizer."}
    )
    save_dir: Optional[str] = field(
        default="../models/trained/incremental_m2g", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    local_dataset_path: Optional[str] = field(
        default="./dataset/validation", metadata={"help": "Path to local dataset."}
    )
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})

    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})

    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})


