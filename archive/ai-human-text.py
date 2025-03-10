import numpy as np
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import gc
import warnings
import requests
import time
import json
warnings.filterwarnings('ignore')

# Constants
CHUNK_SIZE = 10000  # Number of rows to process at a time
MAX_SAMPLES = 50000  # Maximum number of samples to use for training
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
SUPPORTED_LANGUAGES = ['java', 'python', 'cpp']  # Only train on these languages

class CodeDataset(Dataset):
    def __init__(self, code_samples, labels, tokenizer, max_length=256):  # Reduced max_length for optimization
        self.code_samples = code_samples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.code_samples)
    
    def __getitem__(self, idx):
        code = str(self.code_samples[idx])
        label = self.labels[idx]
        
        # Tokenize the code
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def clean_code(code):
    """Clean and normalize code."""
    if not isinstance(code, str):
        return ""
    
    # Remove comments
    code = re.sub(r'#.*', '', code)  # Python comments
    code = re.sub(r'//.*', '', code)  # C/C++/Java comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
    
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    
    # Truncate very long code samples for efficiency
    if len(code) > 5000:
        code = code[:5000]
    
    return code

def get_language(filename):
    """Extract programming language from filename extension."""
    if not isinstance(filename, str):
        return "unknown"
    
    ext = filename.split('.')[-1].lower()
    language_map = {
        'py': 'python',
        'java': 'java',
        'cpp': 'cpp'
    }
    return language_map.get(ext, "unknown")

def process_chunk(chunk):
    """Process a chunk of the dataset."""
    # Extract languages
    chunk['language'] = chunk['file'].apply(get_language)
    
    # Filter only supported languages
    chunk = chunk[chunk['language'].isin(SUPPORTED_LANGUAGES)]
    
    # Clean code
    chunk['cleaned_code'] = chunk['flines'].apply(clean_code)
    
    # Filter out empty code
    chunk = chunk[chunk['cleaned_code'].str.len() > 0]
    
    return chunk[['language', 'cleaned_code']]

def check_internet_connection():
    """Check if we have an active internet connection."""
    try:
        # Try to connect to the Hugging Face website
        requests.get('https://huggingface.co', timeout=5)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

def download_model_with_retry(model_name, tokenizer_class, model_class, num_retries=3, **kwargs):
    """Try to download a model with retries."""
    for i in range(num_retries):
        try:
            print(f"Attempt {i+1}/{num_retries} to download {model_name}...")
            # Set cache directory explicitly
            cache_dir = '/kaggle/working/model_cache'
            os.makedirs(cache_dir, exist_ok=True)
            
            print("Downloading tokenizer...")
            tokenizer = tokenizer_class.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False  # Force download from HF
            )
            
            print("Downloading model...")
            model = model_class.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False,  # Force download from HF
                **kwargs
            )
            
            print(f"✅ Successfully downloaded {model_name}!")
            return tokenizer, model
        except Exception as e:
            print(f"❌ Attempt {i+1} failed: {str(e)}")
            if i < num_retries - 1:
                wait_time = 2 ** i  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"All {num_retries} attempts failed. Falling back to alternative model.")
                return None, None

def create_simple_tokenizer_and_model():
    """Create a simple tokenizer and model for offline use."""
    print("Creating simple tokenizer and model for offline use...")
    
    # Simple tokenizer class
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"[PAD]": 0, "[UNK]": 1}
            self.max_len = 256
        
        def __call__(self, text, max_length=256, padding='max_length', truncation=True, return_tensors='pt'):
            # Simple tokenization by splitting on whitespace
            tokens = text.split()
            
            # Convert tokens to IDs
            ids = []
            for token in tokens[:max_length-2]:  # Leave room for special tokens
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    # Add to vocab if new
                    if len(self.vocab) < 10000:  # Limit vocab size
                        self.vocab[token] = len(self.vocab)
                        ids.append(self.vocab[token])
                    else:
                        ids.append(self.vocab["[UNK]"])
            
            # Pad to max_length
            if padding == 'max_length':
                ids = ids + [self.vocab["[PAD]"]] * (max_length - len(ids))
            
            # Convert to tensor
            if return_tensors == 'pt':
                input_ids = torch.tensor([ids])
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
    
    # Simple model class
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size=10000, num_labels=2):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, 64)
            self.lstm = torch.nn.LSTM(64, 128, batch_first=True)
            self.classifier = torch.nn.Linear(128, num_labels)
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
        def forward(self, input_ids, attention_mask=None, labels=None):
            # Embedding layer
            embedded = self.embedding(input_ids)
            
            # LSTM layer
            lstm_out, _ = self.lstm(embedded)
            
            # Take the last non-padded output for each sequence
            if attention_mask is not None:
                # Get the last token for each sequence
                last_token_indices = attention_mask.sum(1) - 1
                batch_size = input_ids.shape[0]
                batch_indices = torch.arange(batch_size, device=input_ids.device)
                last_token_hidden = lstm_out[batch_indices, last_token_indices]
            else:
                # Just take the last token
                last_token_hidden = lstm_out[:, -1]
            
            # Classification layer
            logits = self.classifier(last_token_hidden)
            
            # Calculate loss if labels are provided
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            
            return type('ModelOutput', (), {'loss': loss, 'logits': logits})
    
    # Create instances
    tokenizer = SimpleTokenizer()
    model = SimpleModel()
    
    print("Simple tokenizer and model created for offline use.")
    return tokenizer, model

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def generate_synthetic_ai_code(human_code_samples, num_samples=None):
    """Generate synthetic AI-written code based on patterns from human code."""
    if num_samples is None:
        num_samples = len(human_code_samples)
    
    print(f"Generating {num_samples} synthetic AI code samples...")
    ai_code_samples = []
    
    # Common patterns in AI-generated code
    ai_patterns = [
        "# This function",
        "# Function to",
        "# Helper function",
        "# Utility to",
        "# Implementation of",
        "# Efficient implementation",
        "# Time complexity: O(n)",
        "# Space complexity: O(1)",
    ]
    
    # Common variable naming patterns in AI code
    ai_variable_patterns = [
        "result", "output", "data", "input_data", "arr", "array", 
        "nums", "numbers", "string", "text", "value", "count"
    ]
    
    import random
    import string
    
    for i in range(num_samples):
        # Select a random human code sample as base
        idx = random.randint(0, len(human_code_samples) - 1)
        base_code = human_code_samples[idx]
        
        if not isinstance(base_code, str):
            base_code = "def empty_function():\n    pass"
        
        # Apply transformations to make it look AI-generated
        ai_code = base_code
        
        # 1. Add AI-style comments
        num_comments = random.randint(1, 3)
        for _ in range(num_comments):
            comment = random.choice(ai_patterns)
            ai_code = comment + "\n" + ai_code
        
        # 2. Rename some variables to typical AI variable names
        for var_name in ai_variable_patterns:
            if random.random() < 0.3:  # 30% chance to replace
                random_var = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))
                ai_code = ai_code.replace(" " + random_var + " ", " " + var_name + " ")
        
        # 3. Add docstring if not present
        if '"""' not in ai_code and "'''" not in ai_code:
            if random.random() < 0.7:  # 70% chance to add docstring
                function_name = "function"
                if "def " in ai_code:
                    parts = ai_code.split("def ")[1].split("(")[0].strip()
                    if parts:
                        function_name = parts
                
                docstring = f'    """\n    This {function_name} performs the required operation.\n    \n    Args:\n        param1: The first parameter\n    \n    Returns:\n        The result of the operation\n    """\n'
                
                if "def " in ai_code and ":" in ai_code:
                    insert_pos = ai_code.find(":", ai_code.find("def ")) + 1
                    ai_code = ai_code[:insert_pos] + "\n" + docstring + ai_code[insert_pos:]
        
        # 4. Add type hints randomly
        if random.random() < 0.5 and "def " in ai_code:  # 50% chance to add type hints
            ai_code = ai_code.replace("(", "(input_data: list) -> ", 1)
        
        ai_code_samples.append(ai_code)
    
    return ai_code_samples

def main():
    try:
        print("Starting code classification training...")
        
        # Create all necessary directories at the start
        working_dir = '/kaggle/working'
        model_cache_dir = os.path.join(working_dir, 'model_cache')
        best_model_dir = os.path.join(working_dir, 'best_model')
        
        for directory in [working_dir, model_cache_dir, best_model_dir]:
            ensure_directory_exists(directory)
        
        # Check internet connection
        internet_available = check_internet_connection()
        if not internet_available:
            print("⚠️ Warning: Internet connection issues detected. Model download may fail.")
        
        # Initialize tokenizer and model
        model_name = 'distilbert-base-uncased'  # Correct model name without organization prefix
        print(f"Loading model: {model_name}")
        
        if internet_available:
            try:
                # First try loading from cache
                tokenizer = DistilBertTokenizer.from_pretrained(
                    model_name,
                    cache_dir=model_cache_dir
                )
                model = DistilBertForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=model_cache_dir,
                    num_labels=2
                )
                print("✅ Successfully loaded model from cache!")
            except Exception as e:
                print(f"Failed to load from cache: {str(e)}")
                print("Attempting to download model...")
                tokenizer, model = download_model_with_retry(
                    model_name,
                    DistilBertTokenizer,
                    DistilBertForSequenceClassification,
                    num_labels=2
                )
            
            if tokenizer is None or model is None:
                print("❌ Failed to load DistilBERT. Using simple model for offline use.")
                tokenizer, model = create_simple_tokenizer_and_model()
        else:
            print("Using simple model for offline use.")
            tokenizer, model = create_simple_tokenizer_and_model()
        
        # Process dataset in chunks
        print("Processing Google Code Jam 2008 dataset...")
        processed_data = []
        total_samples = 0
        
        # Read and process only the gcj2008.csv file
        file_path = '/kaggle/input/google-code-jam-2008-2022/gcj2008.csv'
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Checking for alternative paths...")
            # Try to find the file in the input directory
            for dirname, _, filenames in os.walk('/kaggle/input'):
                for filename in filenames:
                    if '2008' in filename.lower() and filename.endswith('.csv'):
                        file_path = os.path.join(dirname, filename)
                        print(f"Found alternative file: {file_path}")
                        break
                if file_path != '/kaggle/input/google-code-jam-2008-2022/gcj2008.csv':
                    break
        
        print(f"Reading from: {file_path}")
        
        # Read and process the dataset in chunks
        for chunk in tqdm(pd.read_csv(file_path, chunksize=CHUNK_SIZE), desc="Processing chunks"):
            # Process the chunk
            processed_chunk = process_chunk(chunk)
            
            if len(processed_chunk) > 0:
                processed_data.append(processed_chunk)
                
                total_samples += len(processed_chunk)
                
                # Break if we have enough samples
                if total_samples >= MAX_SAMPLES:
                    break
            
            # Clear memory
            gc.collect()
        
        # Combine processed chunks
        print("Combining processed data...")
        if not processed_data:
            raise ValueError("No valid data found after processing. Check file path and data format.")
            
        df = pd.concat(processed_data, ignore_index=True)
        df = df[:MAX_SAMPLES]  # Limit to max samples
        
        print(f"Language distribution in dataset:")
        print(df['language'].value_counts())
        
        # Create human code samples and labels
        human_code_samples = df['cleaned_code'].values
        human_labels = np.ones(len(human_code_samples))  # 1 for human-written
        
        # Generate synthetic AI code samples
        ai_code_samples = generate_synthetic_ai_code(human_code_samples)
        ai_labels = np.zeros(len(ai_code_samples))  # 0 for AI-generated
        
        # Combine human and AI samples
        all_code_samples = np.concatenate([human_code_samples, ai_code_samples])
        all_labels = np.concatenate([human_labels, ai_labels])
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(len(all_code_samples))
        all_code_samples = all_code_samples[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        print(f"Total dataset size: {len(all_code_samples)} samples")
        print(f"Human samples: {np.sum(all_labels == 1)}")
        print(f"AI samples: {np.sum(all_labels == 0)}")
        
        # Split data
        print("Splitting data into train/validation/test sets...")
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            all_code_samples, all_labels, test_size=0.3, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        # Create datasets
        print("Creating PyTorch datasets...")
        train_dataset = CodeDataset(train_texts, train_labels, tokenizer)
        val_dataset = CodeDataset(val_texts, val_labels, tokenizer)
        test_dataset = CodeDataset(test_texts, test_labels, tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        print("Initializing DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            cache_dir=model_cache_dir,
            num_labels=2  # Binary classification: human vs AI
        )
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        print("Starting training...")
        best_val_accuracy = 0
        
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (Training)'):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (Validation)'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true_labels, val_predictions, average='binary', pos_label=1
            )
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            print(f'Validation Precision: {precision:.4f}')
            print(f'Validation Recall: {recall:.4f}')
            print(f'Validation F1: {f1:.4f}')
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print("Saving best model...")
                try:
                    # Ensure the directory exists before saving
                    ensure_directory_exists(best_model_dir)
                    
                    if hasattr(model, 'save_pretrained'):
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        print(f"Saved model and tokenizer to {best_model_dir}")
                    else:
                        # Simple model doesn't have save_pretrained
                        model_path = os.path.join(best_model_dir, 'model.pt')
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved model state dictionary to {model_path}")
                except Exception as e:
                    print(f"Warning: Failed to save model: {str(e)}")
                    print("Continuing training without saving...")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        model.eval()
        test_predictions = []
        test_true_labels = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                human_probs = probs[:, 1].cpu().numpy()  # Probability of being human-written
                
                predictions = torch.argmax(outputs.logits, dim=1)
                test_predictions.extend(predictions.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())
                test_probabilities.extend(human_probs)
        
        # Calculate and save test metrics
        test_accuracy = accuracy_score(test_true_labels, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true_labels, test_predictions, average='binary', pos_label=1
        )
        conf_matrix = confusion_matrix(test_true_labels, test_predictions)
        
        # Save some example predictions with probabilities
        examples = []
        for i in range(min(10, len(test_texts))):
            examples.append({
                'code': test_texts[i][:100] + "..." if len(test_texts[i]) > 100 else test_texts[i],
                'true_label': 'Human' if test_true_labels[i] == 1 else 'AI',
                'predicted_label': 'Human' if test_predictions[i] == 1 else 'AI',
                'human_probability': float(test_probabilities[i])
            })
        
        test_results = {
            'accuracy': float(test_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'examples': examples
        }
        
        # Save test results
        try:
            results_path = os.path.join(working_dir, 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=4)
            print(f"\nSaved test results to {results_path}")
        except Exception as e:
            print(f"Warning: Failed to save test results: {str(e)}")
            print("Test results:", test_results)
        
        print("\nTest Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        print("\nExample Predictions:")
        for i, example in enumerate(examples):
            print(f"Example {i+1}:")
            print(f"  True: {example['true_label']}, Predicted: {example['predicted_label']}, Human Probability: {example['human_probability']:.4f}")
        
        print("\nTraining complete!")
    
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and try again.")

if __name__ == "__main__":
    main()

