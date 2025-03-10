# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="Captain-1337/CrudeBERT")

#   ---------------- OR -------------------

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Captain-1337/CrudeBERT")
model = AutoModelForSequenceClassification.from_pretrained("Captain-1337/CrudeBERT")
