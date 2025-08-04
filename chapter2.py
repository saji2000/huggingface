import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "./bert-model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = ["I've been waiting for a HuggingFace course my whole life.", "Me too"]

tokens = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

print(output)
# input_ids = torch.tensor([ids])
# print("Input IDs:", input_ids)

# output = model(input_ids)
# print("Logits:", output.logits)