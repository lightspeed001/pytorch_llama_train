# Save the model and tokenizer
model.save_pretrained("./tinyllama_finetuned")
tokenizer.save_pretrained("./tinyllama_finetuned")

# Load the saved model for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
finetuned_model = PeftModel.from_pretrained(base_model, "./tinyllama_finetuned")

# Test inference
input_text = "To be or not to be"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = finetuned_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

