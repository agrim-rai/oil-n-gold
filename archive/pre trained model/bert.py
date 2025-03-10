from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "pritamdeb68/BERTAIDetector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_ai_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    ai_probability = probabilities[0][1].item()
    return ai_probability

# Example usage
code_snippet = "def hello_world():\n    print('Hello, World!')"
ai_probability = detect_ai_code(code_snippet)
print(f"Probability of AI-generated code: {ai_probability:.2f}")
