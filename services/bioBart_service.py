from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("GanjinZero/biobart-base")


from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

path = "documents/harry.txt"
text = Path(path).read_text(encoding="utf-8")

print(text)

res = summarizer(
    text,
    max_new_tokens=200,
    truncation=True
)
print(res[0]["summary_text"])
