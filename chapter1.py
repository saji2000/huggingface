from transformers import pipeline

# Create the text-generation pipeline, explicitly specifying the model
classifier = pipeline("text-generation", model="openai-community/gpt2")

# Generate text and store the result
generated_text = classifier("Hi, how are you?", max_length=50)

# Print the generated text
print(generated_text[0]["generated_text"])