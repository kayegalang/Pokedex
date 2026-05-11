import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Load data
def load_json(path):
    with open(path) as f:
        return json.load(f)

train_data = load_json("train.json")
val_data = load_json("val.json")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Format each example into a prompt
def format_example(example):
    text = f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}</s>"
    return {"text": text}

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Formatting and tokenizing dataset...")
train_dataset = train_dataset.map(format_example).map(tokenize)
val_dataset = val_dataset.map(format_example).map(tokenize)

# Training arguments
training_args = TrainingArguments(
    output_dir="./pokedex-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

model.save_pretrained("./pokedex-lora")
tokenizer.save_pretrained("./pokedex-lora")
print("Done! Model saved to ./pokedex-lora")