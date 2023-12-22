from trl import SFTTrainer
from lora_config import peft_parameters
from train_params import training_args
from get_model import model, tokenizer
from get_dataset import dataset

def get_trainer(model, training_args, tokenizer):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=peft_parameters,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )
    return trainer