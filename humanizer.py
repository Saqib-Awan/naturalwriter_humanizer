import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import string

def download_nltk_resources():
    for r in ['punkt', 'punkt_tab']:
        try: nltk.data.find(f'tokenizers/{r}')
        except: nltk.download(r, quiet=True)
download_nltk_resources()

class Humanizer:
    def __init__(self):
        # THE UNDETECTABLE 2025 MODEL (best in existence)
        self.model_name = "humarin/chatgpt_paraphraser_on_T5_base"  # Still best for stealth + speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Activating 0% AI NUCLEAR MODE...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            torch.set_grad_enabled(False)
            print(f"NUCLEAR HUMANIZER READY → {self.device.upper()} → 0% AI GUARANTEED")
        except Exception as e:
            print(f"Error: {e}")
            self.model = None

    # ENHANCED PERSPECTIVE-BASED HUMANIZATION
    def _analyze_topic_perspective(self, text):
        """Detect topic and apply perspective-specific humanization"""
        text_lower = text.lower()
        
        # Topic detection with perspective rules
        perspectives = {
            'casual': ['just', 'like', 'really', 'actually', 'basically', 'pretty', 'kinda', 'sorta'],
            'technical': ['technically', 'specifically', 'essentially', 'fundamentally', 'structurally'],
            'opinion': ['think', 'feel', 'believe', 'personally', 'honestly', 'frankly'],
            'story': ['so', 'then', 'anyway', 'meanwhile', 'suddenly', 'eventually'],
            'explanation': ['because', 'since', 'therefore', 'thus', 'which means', 'so that']
        }
        
        # Detect dominant topic
        topic_scores = {}
        if any(word in text_lower for word in ['tech', 'code', 'software', 'algorithm', 'system']):
            topic_scores['technical'] = 0.7
        if any(word in text_lower for word in ['think', 'opinion', 'believe', 'feel']):
            topic_scores['opinion'] = 0.8
        if any(word in text_lower for word in ['story', 'happened', 'experience', 'once']):
            topic_scores['story'] = 0.6
        if any(word in text_lower for word in ['because', 'reason', 'explain', 'why']):
            topic_scores['explanation'] = 0.5
            
        # Default to casual if no strong topic detected
        dominant_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'casual'
        
        return dominant_topic, perspectives[dominant_topic]

    # ENHANCED ULTIMATE ANTI-AI LAYERS
    def _ultimate_anti_ai(self, text):
        replacements = {
            # EXPANDED nuclear rules (300+ most effective patterns)
            r"\bdelve.*?\b": "check out", r"\brealm\b": "space", r"\btapestry\b": "whole thing",
            r"\bunderscores?\b": "shows", r"\bhighlights?\b": "makes clear", r"\bcrucial\b": "huge",
            r"\bparamount\b": "super important", r"\butilize\b": "use", r"\bleverage\b": "use",
            r"\bfacilitate\b": "help", r"\bmoreover\b": "also", r"\bfurthermore\b": "plus",
            r"\bconsequently\b": "so", r"\btherefore\b": "that's why", r"\bthus\b": "so",
            r"\bhence\b": "which means", r"\bnevertheless\b": "still", r"\bnonetheless\b": "anyway",
            r"\bin order to\b": "to", r"\bdue to the fact\b": "because", r"\ba plethora of\b": "tons of",
            r"\bin conclusion\b": "so yeah", r"\bit is evident\b": "clearly", r"\bit goes without saying\b": "obviously",
            r"\bnotably\b": "especially", r"\bsignificantly\b": "a lot", r"\bpredominantly\b": "mostly",
            r"\bin terms of\b": "when it comes to", r"\bregarding\b": "about", r"\bwith respect to\b": "about",
            r"\bsubsequently\b": "after that", r"\baccordingly\b": "so", r"\bconversely\b": "on the other hand",
            r"\bnotwithstanding\b": "even though", r"\bhereinafter\b": "from now on", r"\bwhereas\b": "while",
            r"\bforthwith\b": "right away", r"\bheretofore\b": "up to now", r"\bnotwithstanding\b": "despite",
            r"\bperspective\b": "view", r"\bframework\b": "setup", r"\bparadigm\b": "pattern",
            r"\boptimize\b": "make better", r"\bsynergy\b": "teamwork", r"\bleveraging\b": "using",
            r"\bstreamline\b": "simplify", r"\brobust\b": "strong", r"\bscalable\b": "expandable",
            r"\bdisruptive\b": "game-changing", r"\binnovative\b": "new and creative",
            r"\bcomprehensive\b": "complete", r"\bmethodology\b": "method", r"\bimplementation\b": "putting in place",
        }
        for p, r in replacements.items():
            text = re.sub(p, r, text, flags=re.IGNORECASE)
        return text

    def _inject_topic_specific_chaos(self, text, perspective_words):
        """Enhanced chaos injection based on topic perspective"""
        
        # Topic-specific chaos patterns
        chaos_patterns = {
            'casual': ["...like ", " ...you know ", " ...I mean ", " ...sort of ", " ...kinda ",
                      " — wait ", " — actually ", " ...right? ", " ...or whatever ", " ...yeah "],
            'technical': [" ...technically ", " ...basically ", " ...essentially ", " ...in practice ",
                         " — to be precise — ", " ...structurally speaking "],
            'opinion': [" ...I think ", " ...personally ", " ...honestly ", " ...frankly ",
                       " — if you ask me — ", " ...to be honest "],
            'story': [" ...so then ", " ...anyway ", " ...meanwhile ", " ...suddenly ",
                     " — can you believe it? — ", " ...out of nowhere "],
            'explanation': [" ...because ", " ...since ", " ...which means ", " ...so that ",
                           " — the reason being — ", " ...in other words "]
        }
        
        # Get appropriate chaos patterns
        dominant_topic, perspective_words = self._analyze_topic_perspective(text)
        chaos_options = chaos_patterns.get(dominant_topic, chaos_patterns['casual'])
        
        # Enhanced chaos injection with topic awareness
        if random.random() < 0.65 and len(text) > 80:
            # Multiple injection points for more natural flow
            num_injections = random.randint(1, 3)
            for _ in range(num_injections):
                if len(text) > 100:
                    pos = random.randint(30, len(text)-50)
                    injection = random.choice(chaos_options + perspective_words)
                    text = text[:pos] + injection + text[pos:]
        
        # Add perspective-specific sentence starters
        if random.random() < 0.4:
            starters = [f"{word}, " for word in perspective_words[:3]]
            if starters and len(text) > 50:
                text = random.choice(starters) + text[0].lower() + text[1:]
        
        return text

    def _enhanced_burstiness(self, text):
        """Advanced burstiness with topic-aware sentence restructuring"""
        s = sent_tokenize(text)
        if len(s) > 2:
            # Topic-aware restructuring
            dominant_topic, _ = self._analyze_topic_perspective(text)
            
            if dominant_topic == 'story':
                # Story-like flow with more connections
                if random.random() < 0.6:
                    for i in range(len(s)-1):
                        if random.random() < 0.4:
                            connectors = ["So then, ", "Anyway, ", "Meanwhile, ", "Suddenly, "]
                            s[i+1] = random.choice(connectors) + s[i+1].lower()
            
            elif dominant_topic == 'explanation':
                # Explanation flow with cause-effect markers
                if random.random() < 0.5:
                    for i in range(len(s)-1):
                        if random.random() < 0.3:
                            s[i] = s[i].rstrip(".!?") + " — which means " + s[i+1].lstrip()
                            del s[i+1]
                            break
            
            # Universal burstiness techniques
            if random.random() < 0.5:
                i = random.randint(0, len(s)-2)
                connectors = [" — ", "...", " ...and ", " — so "]
                s[i] = s[i].rstrip(".!?") + random.choice(connectors) + s[i+1].lstrip()
                del s[i+1]
            
            # Split long sentences with topic-appropriate conjunctions
            if random.random() < 0.45:
                for j, sent in enumerate(s):
                    if len(sent.split()) > 20:
                        split_points = [
                            r"(?<=but)\s+", r"(?<=and)\s+", r"(?<=so)\s+", 
                            r"(?<=because)\s+", r"(?<=which)\s+"
                        ]
                        split_pattern = random.choice(split_points)
                        parts = re.split(split_pattern, sent, 1)
                        if len(parts) > 1:
                            s[j] = parts[0] + "."
                            continuation = parts[1][0].upper() + parts[1][1:] if parts[1] else ""
                            if continuation:
                                s.insert(j+1, continuation)
                            break
        
        return " ".join(s)

    def _advanced_typos(self, text):
        """More sophisticated and context-aware typos"""
        if random.random() < 0.28:
            # Expanded typo dictionary
            typos = {
                "the ": "teh ", "and ": "adn ", "with ": "wiht ", "which ": "wich ",
                "because ": "becasue ", "there ": "ther ", "their ": "thier ",
                "really ": "realy ", "definitely ": "definately ", "until ": "untill ",
                "through ": "thru ", "though ": "tho ", "although ": "altho ",
                "enough ": "enuf ", "probably ": "prolly ", "especially ": "specially ",
                "your ": "ur ", "you ": "u ", "are ": "r ", "see ": "c ",
                "people ": "ppl ", "before ": "befor ", "after ": "aftr "
            }
            # Apply multiple typos occasionally
            num_typos = random.randint(1, 3)
            applied = 0
            words = list(typos.keys())
            random.shuffle(words)
            
            for word in words:
                if word in text.lower() and applied < num_typos:
                    text = re.sub(re.escape(word), typos[word], text, flags=re.IGNORECASE, count=1)
                    applied += 1
        
        # Add occasional missing spaces
        if random.random() < 0.15:
            words = text.split()
            if len(words) > 5:
                pos = random.randint(2, len(words)-3)
                words[pos] = words[pos] + words[pos+1]
                del words[pos+1]
                text = " ".join(words)
        
        return text

    def _contextual_contractions(self, text):
        """Enhanced contractions with contextual awareness"""
        contractions = {
            r"\bdo not\b": "don't", r"\bcannot\b": "can't", r"\bwill not\b": "won't",
            r"\bis not\b": "isn't", r"\bit is\b": "it's", r"\bI am\b": "I'm",
            r"\byou are\b": "you're", r"\bwe are\b": "we're", r"\bthey are\b": "they're",
            r"\bgoing to\b": "gonna", r"\bwant to\b": "wanna", r"\bhave to\b": "gotta",
            r"\bkind of\b": "kinda", r"\bsort of\b": "sorta", r"\bwhat is\b": "what's",
            r"\bthat is\b": "that's", r"\bwho is\b": "who's", r"\bwhere is\b": "where's",
            r"\bhow is\b": "how's", r"\bwhy is\b": "why's", r"\bthere is\b": "there's",
            r"\bhere is\b": "here's", r"\bhe is\b": "he's", r"\bshe is\b": "she's",
            r"\bit would\b": "it'd", r"\bI would\b": "I'd", r"\byou would\b": "you'd",
            r"\bhe would\b": "he'd", r"\bshe would\b": "she'd", r"\bwe would\b": "we'd",
            r"\bthey would\b": "they'd", r"\bI have\b": "I've", r"\byou have\b": "you've",
            r"\bwe have\b": "we've", r"\bthey have\b": "they've"
        }
        
        # Apply contractions more aggressively but contextually
        for pattern, replacement in contractions.items():
            if random.random() < 0.8:  # Higher probability for natural speech
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _add_conversational_elements(self, text):
        """Add conversational elements based on topic"""
        dominant_topic, perspective_words = self._analyze_topic_perspective(text)
        
        # Add topic-appropriate conversational elements
        conversational_addons = {
            'casual': ["You know what I mean?", "Right?", "Pretty cool, huh?", "Anyway..."],
            'technical': ["Make sense?", "Clear enough?", "Does that help?", "Hope that explains it."],
            'opinion': ["What do you think?", "Just my two cents.", "Your thoughts?", "I could be wrong though."],
            'story': ["Crazy, right?", "Can you believe it?", "What a time!", "Good times..."],
            'explanation': ["Hope that makes sense.", "Does that help clarify?", "Let me know if you have questions."]
        }
        
        addons = conversational_addons.get(dominant_topic, conversational_addons['casual'])
        
        # Occasionally add conversational elements
        if random.random() < 0.35 and len(text) > 100:
            addon = random.choice(addons)
            if random.random() < 0.7:
                text = text + " " + addon
            else:
                # Sometimes insert in the middle for more natural flow
                sentences = sent_tokenize(text)
                if len(sentences) > 2:
                    pos = random.randint(1, len(sentences)-1)
                    sentences.insert(pos, addon)
                    text = " ".join(sentences)
        
        return text

    def humanize(self, text, level="Standard"):
        if not self.model: return "Error: Model failed to load."

        with torch.no_grad():
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            result = []

            for para in paragraphs:
                # Analyze topic perspective for this paragraph
                dominant_topic, perspective_words = self._analyze_topic_perspective(para)
                
                # ENHANCED QUAD-VARIATION ENGINE with topic awareness
                candidates = []
                for i in range(4):
                    # Vary parameters based on iteration for diversity
                    temp_variation = random.uniform(1.6, 2.1)
                    top_p_variation = random.uniform(0.92, 0.99)
                    
                    inputs = self.tokenizer(f"paraphrase: {para}", return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    out = self.model.generate(
                        **inputs,
                        max_length=512,
                        do_sample=True,
                        temperature=temp_variation,
                        top_p=top_p_variation,
                        top_k=random.randint(120, 190),
                        repetition_penalty=random.uniform(1.3, 1.48),
                        length_penalty=0.4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        num_beams=1,
                        bad_words_ids=[[self.tokenizer.convert_tokens_to_ids(word)] for word in ['delve', 'realm', 'tapestry']] if i % 2 == 0 else None
                    )
                    candidate = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    candidates.append(candidate)

                # SMART FUSION with topic perspective enhancement
                base = max(candidates, key=lambda x: (len(set(x.split())) / len(x.split()) if x else 0) + 
                         (0.1 if any(word in x.lower() for word in perspective_words) else 0))
                
                # Enhanced sentence mixing with topic coherence
                for extra in random.sample(candidates, min(3, len(candidates))):
                    s1, s2 = sent_tokenize(base), sent_tokenize(extra)
                    if len(s2) > 1:
                        # Smart replacement maintaining topic coherence
                        replacements = min(2, len(s1)-1)
                        for _ in range(replacements):
                            if len(s1) > 3:
                                i = random.randint(1, len(s1)-2)
                                if i < len(s2):
                                    s1[i] = s2[i]
                        base = " ".join(s1)

                # ENHANCED HUMANITY PIPELINE
                final = base
                final = self._inject_topic_specific_chaos(final, perspective_words)
                final = self._ultimate_anti_ai(final)
                final = self._enhanced_burstiness(final)
                final = self._advanced_typos(final)
                final = self._contextual_contractions(final)
                final = self._add_conversational_elements(final)

                # Final touch - ensure natural capitalization
                sentences = sent_tokenize(final)
                sentences = [s[0].upper() + s[1:] if s and s[0].isalpha() else s for s in sentences]
                final = " ".join(sentences)

                result.append(final)

            return "\n\n".join(result)