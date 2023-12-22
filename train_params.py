from transformers import TrainingArguments

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