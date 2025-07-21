from transformers import pipeline

# Initialize the translation pipeline with a different model
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fa", use_fast=True)

# Example translation
text = "What is your name?"
translated = translator(text)
print(translated)