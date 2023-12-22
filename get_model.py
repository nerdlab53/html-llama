from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import quantization_config

def get_model(model_name : str, trust_remote_code : bool = True):
    model = AutoModelForCausalLM.from_pretrained(model_name, \
        quantization_config=quantization_config, \
        device_map={"":0})
    model.config.use_cache = False
    model.config.pretraining_tp = 1 
    tokenizer = AutoTokenizer.from_pretrained(model_name, \
        trust_remote_code=trust_remote_code)
    return model, tokenizer