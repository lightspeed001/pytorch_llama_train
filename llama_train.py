# Training TinyLlama-1.1B on Colab

# Below pip is only for Kaggle or Colab notebooks
#!pip install -q transformers accelerate datasets peft bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 8-bit quantization and auto device mapping
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype=torch.float16, # use mixed precision
  device_map="auto", # automatically use available GPU
  load_in_8bit=True, # Use 8-bit quantisation
)

# Apply LoRA
lora_config = LoraConfig(
  r=8, # rank of low-rank matrices (smaller = less memory)
  lora_alpha=32, # scaling factor for LoRA weights
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention layers
  lora_dropout=0.05, # dropout for regularization
  bias="none", # no bias terms
  task_type="CAUSAL_LM" # for causal language modeling
)

#Apply LoRA to model
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")
#model.print_trainable_parameters()  # Check how many parameters are trainable

# Load dataset (example: tiny_shakespeare)

#from datasets import load_dataset

#dataset = load_dataset("tiny_shakespeare", split="train").map(
#  lambda x: tokenizer(x["test"], truncation=True, max_length=512),
#  batched=True,
#)

dataset = load_dataset("tiny_shakespeare", split="train")

# Tokenize the dataset
def tokenize_function(examples):
  return tokenizer(
    examples["text"],
    truncation=True,
    max_length=512, # Limit sequence length
    return_overflowing_tokens=True,
    return_length=True,    
  )

# Apply tokenization
tokenized_dataset = dataset.map(
  tokenize_function,
  batched=True,
  remove_columns=["text"], # Remove original text column
)

# Training arguments
training_args = TrainingArguments(
  output_dir="/app/results", # Directory to save checkpoints
  per_device_train_batch_size=4, # Batch size per GPU (Adjust based on GPU memory)
  gradient_accumulation_steps=4, # Simulate larger batch size
  num_train_epochs=1, # Number of epochs
  fp16=True, # Use mixed precision
  save_steps=500, # save checkpoint every 500 steps
  logging_steps=100, # Log metrics every 100 steps
  learning_rate=2e-5, # Learning rate
  weight_decay=0.01, # L2 regulization
  warmup_steps=100, # Learning rate warmup
  optim="paged_adamw_8bit", # Optimizer with 8-bit states
)

# Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset,
)
# Train
trainer.train()

# Save model
model.save_pretrained("/app/results")
tokenizer.save_pretrained("/app/results")
