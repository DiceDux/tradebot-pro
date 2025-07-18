from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# این خطوط فقط یک بار در ابتدای پروژه لازم است (بار اول مدل دانلود می‌شود)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment_finbert(text):
    """
    دریافت عدد سینتیمنت حرفه‌ای برای متن خبر یا تحلیل مالی بین -1 (منفی) تا +1 (مثبت)
    """
    if not text or not isinstance(text, str):
        return 0.0
    try:
        result = finbert_pipe(text[:512])[0]  # FinBERT فقط تا 512 کاراکتر ورودی می‌گیرد
        label = result['label'].lower()
        score = result['score']
        if label == "positive":
            return score  # مثبت
        elif label == "negative":
            return -score  # منفی
        else:  # neutral
            return 0.0
    except Exception as e:
        print(f"[FinBERT Sentiment Error]: {e}")
        return 0.0