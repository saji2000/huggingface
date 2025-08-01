from transformers import AutoTokenizer
import os

# Step 1: Load tokenizer from Hugging Face Hub (or use your custom model name)
tokenizer = AutoTokenizer.from_pretrained("./bert-model")

# Step 2: Save tokenizer locally to the model folder
# tokenizer.from_pretrained("./bert-model")

# Step 3: Print contents of the model directory
print(os.listdir("./bert-model"))

# Step 4: Test encoding a sentence
encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
