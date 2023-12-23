# Fine tuned Llama 2 7B for HTML Code generation

## About
- This is fine tuned Llama 2 which generates HTML code for a given piece of prompt.
- It has been trained using QLoRA (Quantized Low Rank Adaptation) for maximizing the model training performance while keeping the GPU demands reasonably low for training and inference.
- Model chosen for fine-tuning : [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- Trained on retr0sushi04/html_pre_processed can be found on HuggingFace : [dataset](https://huggingface.co/datasets/retr0sushi04/html_pre_processed) which is a preprocessed version of [raw dataset](https://huggingface.co/datasets/jawerty/html_dataset).

## Requirements
- Requirements are listed in requirements.txt and are as follows :
- `torch==2.1.0
  transformers==4.31.0 
  trl==0.4.7 
  bitsandbytes==0.40.2 
  peft==0.4.0 
  accelerate==0.21.0`
- Installation :
- `pip install torch==2.1.0
  transformers==4.31.0 
  trl==0.4.7 
  bitsandbytes==0.40.2 
  peft==0.4.0 
  accelerate==0.21.0`
- **NOTE**: Install torch with CUDA support if using on GPU for even faster training.
  
## Model & Dataset Selection
- Model Selected : [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
  - 7B fine-tuned Llama 2 optimized for dialogue use cases.
- Dataset Selected : [raw dataset](https://huggingface.co/datasets/jawerty/html_dataset)
  - Pre processed and uploaded further to HuggingFace as [dataset](https://huggingface.co/datasets/retr0sushi04/html_pre_processed)
 
## Training Specifications
-  Trained using QLoRA for quantized low precision training in 4-bit for less training time and better performance on the selected dataset.
-  Used BitsAndBytes for loading the quantized model from HuggingFace.
-  Bits and Bytes config is given below :
   ```Python
   from transformers import BitsAndBytesConfig
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
   ```
## Evals

## Inference 

## Imporvements and possible challenges
