from transformers import LoraConfig

# lora config hyperparams
lora_alpha = 16
lora_dropout = 0.1
r = 64
bias = "none"
task_type = "CAUSAL_LM"

peft_parameters = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=r,
    bias=bias,
    task_type=task_type
)