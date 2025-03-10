import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from datasets import load_dataset

# Load CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Function to clean code (removes comments, extra spaces)
def clean_code(code):
    if not isinstance(code, str):
        return ""
    code = re.sub(r"#.*", "", code)  # Remove Python comments
    code = re.sub(r"//.*", "", code)  # Remove C++/Java comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)  # Remove block comments
    code = re.sub(r"\s+", " ", code).strip()  # Remove extra spaces
    return code

# Function to normalize variable and function names
def normalize_code(code):
    return re.sub(r"\b[a-zA-Z_]\w*\b", "VAR", code)  # Replace variable names

# Function to tokenize code
def tokenize_code(code):
    return tokenizer(code, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Load datasets
def load_datasets():
    print("Loading datasets...")
    
    try:
        # Load datasets directly from Hugging Face
        print("Loading OpenAI HumanEval dataset from Hugging Face...")
        humaneval_dataset = load_dataset("openai/humaneval")
        
        print("Loading Codeforces Python Submissions dataset from Hugging Face...")
        codeforces_dataset = load_dataset("MatrixStudio/Codeforces-Python-Submissions")
        
        # Extract problem statements and solutions from HumanEval
        humaneval_prompts = []
        humaneval_solutions = []
        
        for item in humaneval_dataset["test"]:  # Using test split as it contains all examples
            if "prompt" in item and "canonical_solution" in item:
                humaneval_prompts.append(item["prompt"])
                humaneval_solutions.append(item["canonical_solution"])
        
        print(f"Extracted {len(humaneval_prompts)} problems from HumanEval dataset")
        
        # Extract problem statements and solutions from Codeforces
        codeforces_prompts = []
        codeforces_solutions = []
        
        for item in codeforces_dataset["train"]:  # Using train split
            problem_desc = ""
            if "problem-description" in item and item["problem-description"]:
                problem_desc = item["problem-description"]
            elif "title" in item and item["title"]:
                problem_desc = item["title"]
                
            if problem_desc and "code" in item and item["code"]:
                codeforces_prompts.append(problem_desc)
                codeforces_solutions.append(item["code"])
        
        print(f"Extracted {len(codeforces_prompts)} problems from Codeforces dataset")
        
        # Combine data from both datasets
        problem_statements = codeforces_prompts + humaneval_prompts
        human_solutions = codeforces_solutions + humaneval_solutions
        
        # Limit the number of samples if needed (for faster processing)
        max_samples = 100  # Adjust as needed
        if len(problem_statements) > max_samples:
            print(f"Limiting to {max_samples} samples for faster processing")
            problem_statements = problem_statements[:max_samples]
            human_solutions = human_solutions[:max_samples]
        
        print(f"Total: {len(problem_statements)} problem statements and {len(human_solutions)} human solutions")
        
        return problem_statements, human_solutions
        
    except Exception as e:
        print(f"Error loading datasets from Hugging Face: {e}")
        print("Falling back to sample data...")
        
        # Provide fallback sample data in case of errors
        problem_statements = [
            "Write a function to find the maximum element in a list.",
            "Implement a function to check if a string is a palindrome.",
            "Write a function to calculate the factorial of a number.",
            "Implement a function to find the GCD of two numbers."
        ]
        
        human_solutions = [
            "def find_max(arr):\n    return max(arr)",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"
        ]
        
        print(f"Using {len(problem_statements)} sample problem statements due to error")
        return problem_statements, human_solutions

# Generate AI-generated code using a model from Hugging Face
def generate_ai_code(problem_statements, batch_size=5, max_samples=100):
    print("Generating AI code solutions...")
    
    # Use a smaller model for faster inference
    model_name = "Salesforce/codegen-350M-mono"  # Smaller CodeGen model
    
    try:
        pipe = pipeline("text-generation", model=model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to a smaller model...")
        try:
            model_name = "gpt2"  # Fallback to an even smaller model
            pipe = pipeline("text-generation", model=model_name)
        except Exception as e2:
            print(f"Error loading fallback model {model_name}: {e2}")
            print("Using simple template-based code generation instead.")
            # If we can't load any model, use template-based generation
            return generate_template_code(problem_statements[:max_samples])
    
    ai_solutions = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, min(len(problem_statements), max_samples), batch_size)):
        batch = problem_statements[i:i+batch_size]
        
        for prompt in batch:
            full_prompt = f"Write Python code to solve this problem: {prompt}\n\n"
            
            try:
                generated = pipe(full_prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]
                
                # Extract only the code part (after the prompt)
                code_part = generated[len(full_prompt):].strip()
                
                # If empty or too short, generate a placeholder
                if len(code_part) < 10:
                    code_part = generate_template_solution(prompt)
                
                ai_solutions.append(code_part)
            except Exception as e:
                print(f"Error generating code: {e}")
                # Add a template-based solution on error
                ai_solutions.append(generate_template_solution(prompt))
    
    return ai_solutions

# Generate template-based code for a problem statement
def generate_template_solution(prompt):
    # Extract potential function name from the prompt
    words = prompt.lower().split()
    function_name = "solution"
    
    # Look for keywords that might indicate the function name
    for i, word in enumerate(words):
        if word in ["function", "method"] and i + 2 < len(words):
            potential_name = words[i + 2].strip(".,':;\"")
            if potential_name.isalnum():
                function_name = potential_name
                break
    
    # Create a template solution
    return f"""def {function_name}(input_data):
    # TODO: Implement solution for: {prompt}
    
    # This is an AI-generated placeholder solution
    result = None
    
    # Process the input
    if isinstance(input_data, list):
        # Handle list input
        if len(input_data) > 0:
            result = input_data[0]  # Default to first element
    elif isinstance(input_data, str):
        # Handle string input
        result = input_data  # Default to input string
    elif isinstance(input_data, (int, float)):
        # Handle numeric input
        result = input_data  # Default to input number
    
    return result"""

# Generate template-based code for all problem statements
def generate_template_code(problem_statements):
    print("Using template-based code generation...")
    return [generate_template_solution(prompt) for prompt in problem_statements]

# Custom PyTorch Dataset class
class CodeDataset(Dataset):
    def __init__(self, code_samples, labels):
        self.inputs = []
        self.labels = []
        
        for code, label in zip(code_samples, labels):
            cleaned = clean_code(code)
            normalized = normalize_code(cleaned)
            tokens = tokenize_code(normalized)["input_ids"].squeeze()
            
            # Handle single samples (convert to 1D tensor)
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
                
            self.inputs.append(tokens)
            self.labels.append(label)
            
        self.inputs = torch.stack(self.inputs)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}

# Evaluate model on test data
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating model"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids)
            _, predicted = torch.max(outputs.logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Calculate precision, recall, and F1 score for AI code detection (class 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', pos_label=1
    )
    
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    results = {
        "accuracy": accuracy,
        "precision": precision * 100,  # Convert to percentage
        "recall": recall * 100,        # Convert to percentage
        "f1_score": f1 * 100,          # Convert to percentage
        "confusion_matrix": conf_matrix.tolist()
    }
    
    return results

# Main function
def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Load datasets
    problem_statements, human_solutions = load_datasets()
    
    # Generate AI solutions
    ai_solutions = generate_ai_code(problem_statements)
    
    # Save the processed data
    processed_data = {
        "problem_statement": problem_statements[:len(ai_solutions)],
        "human_solution": human_solutions[:len(ai_solutions)],
        "ai_solution": ai_solutions
    }
    
    # Save as CSV for easier analysis
    df = pd.DataFrame(processed_data)
    df.to_csv("output/processed_solutions.csv", index=False)
    
    print(f"Saved {len(ai_solutions)} processed solutions to output/processed_solutions.csv")
    
    # Prepare datasets for training
    all_code = human_solutions[:len(ai_solutions)] + ai_solutions
    all_labels = [0] * len(ai_solutions) + [1] * len(ai_solutions)  # 0 for human, 1 for AI
    
    # Create dataset
    dataset = CodeDataset(all_code, all_labels)
    
    # Split into train/validation/test sets (70/15/15 split)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")
    
    # Load CodeBERT for classification (2 labels: human, AI)
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Training loop
    num_epochs = 3
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids)
                _, predicted = torch.max(outputs.logits, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained("output/codebert-ai-detector")
            tokenizer.save_pretrained("output/codebert-ai-detector")
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
    
    print("Model training complete! 🚀")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    best_model = RobertaForSequenceClassification.from_pretrained("output/codebert-ai-detector")
    best_model.to(device)
    
    test_results = evaluate_model(best_model, test_loader, device)
    
    print("\nTest Results:")
    print(f"Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision']:.2f}%")
    print(f"Recall: {test_results['recall']:.2f}%")
    print(f"F1 Score: {test_results['f1_score']:.2f}%")
    print("\nConfusion Matrix:")
    print(test_results['confusion_matrix'])
    
    # Save test results
    with open("output/test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    
    print("\nSaved test results to output/test_results.json")

if __name__ == "__main__":
    main()