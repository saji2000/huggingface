# Save as chapter1_gpu.py
from transformers import pipeline
import torch

# Verify GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# Text generation pipeline
classifier = pipeline("text-generation", model="distilgpt2", device=device)
generated_text = classifier("Hi, how are you?", max_new_tokens=50, truncation=True)
print("Generated Text:", generated_text[0]["generated_text"])

# Summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
text = """
America has changed dramatically during recent years. Not only has the number of 
graduates in traditional engineering disciplines such as mechanical, civil, 
electrical, chemical, and aeronautical engineering declined, but in most of 
the premier American universities engineering curricula now concentrate on 
and encourage largely the study of engineering science. As a result, there 
are declining offerings in engineering subjects dealing with infrastructure, 
the environment, and related issues, and greater concentration on high 
technology subjects, largely supporting increasingly complex scientific 
developments. While the latter is important, it should not be at the expense 
of more traditional engineering.

Rapidly developing economies such as China and India, as well as other 
industrial countries in Europe and Asia, continue to encourage and advance 
the teaching of engineering. Both China and India, respectively, graduate 
six and eight times as many traditional engineers as does the United States. 
Other industrial countries at minimum maintain their output, while America 
suffers an increasingly serious decline in the number of engineering graduates 
and a lack of well-educated engineers.
"""
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print("Summary:", summary[0]["summary_text"])