from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingArguments:
    project_name: Optional[str] = field(
        default="", metadata={"help": "Project name."}
    )
    base_model_dir: Optional[str] = field(
        default="../../models/initiialized/gpt2", metadata={"help": "Model name or path of model to be trained."}
    )
    tokenizer_dir: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Path of tokenizer."}
    )
    save_dir: Optional[str] = field(
        default="../../models/trained/centralized_codeparrot", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    local_dataset_path: Optional[str] = field(
        default="../dataset/", metadata={"help": "Path to local dataset."}
    )
    dataset_name_train: Optional[str] = field(
        default="google_train.json.gz", metadata={"help": "Name or path of training dataset."}
    )
    dataset_name_valid: Optional[str] = field(
        default="google_valid.json.gz", metadata={"help": "Name or path of validation dataset."}
    )
    train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=1000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=2000, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=-1, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    max_train_epochs: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum number of train epochs."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})

