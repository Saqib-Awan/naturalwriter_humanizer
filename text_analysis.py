from textblob import TextBlob

class TextAnalyzer:
    def analyze(self, text):
        if not text:
            return {}
            
        blob = TextBlob(text)
        return {
            "word_count": len(text.split()),
            "sentence_count": len(blob.sentences),
            "polarity": round(blob.sentiment.polarity, 2)
        }