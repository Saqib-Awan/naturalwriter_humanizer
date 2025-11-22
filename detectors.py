import math
from textblob import TextBlob

class AIDetector:
    def __init__(self):
        pass

    def detect(self, text):
        """
        Estimates AI probability based on text complexity (Perplexity simulation).
        Since we can't run a 2nd heavy model, we use statistical analysis.
        """
        if not text:
            return {"score": 0, "label": "N/A"}

        words = text.split()
        if len(words) < 5:
            return {"score": 0, "label": "Too Short"}

        # 1. Calculate Sentence Length Variation (Burstiness)
        blob = TextBlob(text)
        sentence_lengths = [len(s.words) for s in blob.sentences]
        
        if not sentence_lengths:
            return {"score": 100, "label": "AI"}

        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        
        # Calculate variance
        variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        
        # AI tends to write with low variance (monotone). Humans vary length.
        # Higher variance = More Human.
        
        # 2. Unique Word Usage (Perplexity proxy)
        unique_ratio = len(set(words)) / len(words)

        # 3. Combine into a 'Human Score'
        # - High variance adds to human score
        # - High unique ratio adds to human score
        
        human_score = min(100, (variance * 1.5) + (unique_ratio * 50))
        
        # Invert for AI Score
        ai_score = 100 - human_score
        
        # Clamp results
        ai_score = max(0, min(100, ai_score))

        label = "Human"
        if ai_score > 60:
            label = "Likely AI"
        elif ai_score > 30:
            label = "Mix"

        return {
            "score": round(ai_score, 1),
            "label": label
        }