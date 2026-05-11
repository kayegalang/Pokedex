import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')

# Load test data
with open("test.json") as f:
    test_data = json.load(f)

# We'll evaluate on one entry per Pokemon to keep it manageable
seen = set()
test_samples = []
for entry in test_data:
    name = entry["input"].replace("Name: ", "")
    if name not in seen:
        seen.add(name)
        test_samples.append(entry)

print(f"Evaluating on {len(test_samples)} Pokemon from test set\n")

def load_model(use_lora=True):
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(
        "./pokedex-lora" if use_lora else base_model_name
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32
    )
    if use_lora:
        model = PeftModel.from_pretrained(base_model, "./pokedex-lora")
    else:
        model = base_model
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, pokemon_name):
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

def evaluate_model(model, tokenizer, label):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

    for entry in test_samples:
        name = entry["input"].replace("Name: ", "")
        reference = entry["output"]
        prediction = generate(model, tokenizer, name)

        # ROUGE
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

        # BLEU
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)

        print(f"  {name}: ROUGE-L={scores['rougeL'].fmeasure:.3f}, BLEU={bleu:.3f}")

    print(f"\n--- {label} Results ---")
    print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
    print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
    print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
    print(f"BLEU:    {np.mean(bleu_scores):.4f}\n")

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
        "bleu": np.mean(bleu_scores)
    }

print("Loading fine-tuned model...")
ft_model, ft_tokenizer = load_model(use_lora=True)
ft_results = evaluate_model(ft_model, ft_tokenizer, "Fine-Tuned Model")

del ft_model
torch.cuda.empty_cache()

print("Loading base model (baseline)...")
base_model, base_tokenizer = load_model(use_lora=False)
base_results = evaluate_model(base_model, base_tokenizer, "Base Model (Baseline)")

print("=== COMPARISON ===")
for metric in ["rouge1", "rouge2", "rougeL", "bleu"]:
    ft_val = ft_results[metric]
    base_val = base_results[metric]
    diff = ft_val - base_val
    print(f"{metric.upper():10} Fine-tuned: {ft_val:.4f}  |  Baseline: {base_val:.4f}  |  Diff: {diff:+.4f}")