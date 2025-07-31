from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./bert-model")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)