from transformers import BitsAndBytesConfig
import torch

# Hyperparameters
# bits and bytes config
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant=False

compute_dtype = getattr(torch, "float16")

quantization_config = BitsAndBytesConfig(
    load_in_4bit = load_in_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
)