import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from huggingface_hub import login
from peft import PeftModel, LoraConfig
from get_dataset import get_dataset
from get_model import get_model
from trainer import get_trainer
import argparse
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    # creating arguments to be passed by the user while running the training script
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-2-7b-chat-hf")
    parser.add_argument("--dataset", type=str, default="retr0sushi04/html")
    parser.add_argument("--hf_repo", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fine_tuned_model_name", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    # loading the model on the GPU
    device_map = {"":0}

    # setting variables from input args
    model_name = args.model_name
    model_repo = args.hf_repo
    new_model = args.fine_tuned_model_name
    dataset = args.dataset
    num_train_epochs = args.epochs
    learning_rate = args.lr

    load_dotenv()

    login(token=os.getenv("HF_TOKEN"))

    compute_dtype = getattr(torch, "float16")
    print('Getting dataset..')
    dataset = get_dataset(dataset)
    print('Got dataset')

    print('Getting model..')
    model, tokenizer = get_model(model_name)
    print('Got model.')

    print('Setting parameters : ')
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
    # Training Parameters
    output_dir="./results"
    num_train_epochs=3
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

    # creating trainer
    trainer = get_trainer(model, training_args, tokenizer)

    # starting training
    print('Starting Traning : ')
    trainer.train()
    print('Training ends..')

    print('Merging weights..')

    
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
    print("----------------------------------")
    print('Pushing model to hub..')
    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)
    print('Model pushed to hub.')