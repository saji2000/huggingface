from transformers import pipeline

# Initialize the translation pipeline with use_fast=True
translator = pipeline("translation", model="abbasmahmudiai/MT5_en_to_persian", use_fast=True)

# Example translation
text = "Hello, how are you?"
translated = translator(text)
print(translated)