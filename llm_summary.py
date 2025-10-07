from transformers import pipeline

# Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-base")

# Example IMDB review text
sample_text = "Moonwalker is part biography, part feature film, part music video. With all this stuff going down at the moment with MJ I've started listening to his music again."

summary = summarizer(sample_text, max_length=60, min_length=15, do_sample=False)
print("Summary:", summary[0]["summary_text"])

# Zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["positive", "negative"]
result = classifier(sample_text, candidate_labels=labels)
print("Zero-shot:", result)
