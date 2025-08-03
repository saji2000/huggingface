import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Step 1: Load tokenizer from Hugging Face Hub (or use your custom model name)
tokenizer = AutoTokenizer.from_pretrained("./bert-model")
model = AutoModelForSequenceClassification.from_pretrained("./bert-model")


# Step 2: Save tokenizer locally to the model folder
# tokenizer.from_pretrained("./bert-model")

# Step 3: Print contents of the model directory
# print(os.listdir("./bert-model"))

# Step 4: Test encoding a sentence
# encoded_input = tokenizer("This is a preloaded model!")
# print(encoded_input)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail.
result = model(input_ids)
print(result)
