from transformers import pipeline

classifier = pipeline("sentiment-analysis")
results = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I love you!",
    ]
)

print(results)