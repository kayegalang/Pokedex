import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./pokedex-lora")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, "./pokedex-lora")
model.eval()

def generate(pokemon_name):
    prompt = f"<|user|>\nName: {pokemon_name}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# Test on a few Pokemon
test_names = ["Pikachu", "Mewtwo", "Snorlax", "Typhlosion", "Lugia", "Togepi"]
for name in test_names:
    print(f"\n{name}:")
    print(generate(name))