import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from huggingface_hub import login
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", type=str, required=True)
parser.add_argument("--hf-repo", type = str, required=True)
args = parser.parse_args()

device_map = {"":0}

login(token=args.hf_token)

new_model = args.hf_repo

# Hyperparameters
# bits and bytes config
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant=False

# lora config hyperparams
lora_alpha = 16
lora_dropout = 0.1
r = 64
bias = "none"
task_type = "CAUSAL_LM"

# Training Parameters
output_dir="./results"
num_train_epochs=1
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=1
gradient_checkpointing=True
optim="paged_adamw_32bit"
save_steps=0
logging_steps=1
evaluation_strategy="steps"
learning_rate=2e-4
weight_decay=0.001
fp16=False
bf16=False
max_grad_norm=0.3
max_steps=-1
warmup_ratio=0.03
group_by_length=True
lr_scheduler_type="cosine"
report_to="tensorboard"

print("Loading dataset..")
dataset = load_dataset("retr0sushi04/html", split = "train")
dataset = dataset.train_test_split(test_size=0.2)
print("Dataset loaded")

print("Loading model..")
model_name = "meta-llama/Llama-2-7b-chat-hf"
print("Model loaded")
compute_dtype = getattr(torch, "float16")
quantization_config = BitsAndBytesConfig(
    load_in_4bit = load_in_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
)

from huggingface_hub import notebook_login
notebook_login()

# loading model onto CPU(GPU if available)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"":0})

model.config.use_cache = False
model.config.pretraining_tp = 1

# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_parameters = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=r,
    bias=bias,
    task_type=task_type
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy=evaluation_strategy,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to
  )

# creating a supervised finetuning trainer
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
print("Beginning training..")
trainer.train()
print("Training over")

trainer.model.save_pretrained(new_model)

del model
del trainer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()
gc.collect()


print("Merging weights")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map
)

model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()



tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Pushing to hub..")
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
print("Pushed to hub.")