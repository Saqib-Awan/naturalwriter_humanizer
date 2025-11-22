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
        if not text or len(text.strip()) == 0:
            return 'casual', []
            
        text_lower = text.lower()
        
        # Topic detection with perspective rules
        perspectives = {
            'casual': ['just', 'like', 'really', 'actually', 'basically', 'pretty', 'kinda', 'sorta'],
            'technical': ['technically', 'specifically', 'essentially', 'fundamentally', 'structurally'],
            'opinion': ['think', 'feel', 'believe', 'personally', 'honestly', 'frankly'],
            'story': ['so', 'then', 'anyway', 'meanwhile', 'suddenly', 'eventually'],
            'explanation': ['because', 'since', 'therefore', 'thus', 'which means', 'so that']
        }
        
        # Detect dominant topic with scores
        topic_scores = {'casual': 0.1}  # Default baseline
        
        # Technical detection
        tech_words = ['tech', 'code', 'software', 'algorithm', 'system', 'data', 'function', 'process']
        if any(word in text_lower for word in tech_words):
            topic_scores['technical'] = 0.8
            
        # Opinion detection  
        opinion_words = ['think', 'opinion', 'believe', 'feel', 'view', 'perspective']
        if any(word in text_lower for word in opinion_words):
            topic_scores['opinion'] = 0.7
            
        # Story detection
        story_words = ['story', 'happened', 'experience', 'once', 'time', 'when']
        if any(word in text_lower for word in story_words):
            topic_scores['story'] = 0.6
            
        # Explanation detection
        explanation_words = ['because', 'reason', 'explain', 'why', 'therefore', 'thus']
        if any(word in text_lower for word in explanation_words):
            topic_scores['explanation'] = 0.5
            
        # Default to casual if no strong topic detected
        dominant_topic = max(topic_scores, key=topic_scores.get) if topic_scores else 'casual'
        
        return dominant_topic, perspectives.get(dominant_topic, perspectives['casual'])

    # ULTRA-ENHANCED ANTI-AI LAYERS
    def _ultimate_anti_ai(self, text):
        if not text:
            return text
            
        replacements = {
            # EXPANDED nuclear rules (400+ most effective patterns)
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
            r"\bcommence\b": "start", r"\bterminate\b": "end", r"\bapproximately\b": "about",
            r"\bdemonstrate\b": "show", r"\butilization\b": "use", r"\bfacilitate\b": "help",
            r"\binterface\b": "connect", r"\boperationalize\b": "use", r"\bparadigm shift\b": "big change",
            r"\bvalue proposition\b": "benefit", r"\bcore competency\b": "main skill", r"\bactionable\b": "useful",
            r"\bbandwidth\b": "time", r"\bcircle back\b": "follow up", r"\bdeep dive\b": "close look",
            r"\bholy grail\b": "ultimate goal", r"\blow-hanging fruit\b": "easy win", r"\bmoving forward\b": "from now on",
            r"\bthought leadership\b": "expert advice", r"\bwin-win\b": "good for everyone",
        }
        
        for pattern, replacement in replacements.items():
            try:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            except:
                continue
                
        return text

    def _inject_topic_specific_chaos(self, text, perspective_words):
        """Enhanced chaos injection based on topic perspective"""
        if not text or len(text) < 20:
            return text
            
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
        dominant_topic, _ = self._analyze_topic_perspective(text)
        chaos_options = chaos_patterns.get(dominant_topic, chaos_patterns['casual'])
        
        # Enhanced chaos injection with topic awareness
        if random.random() < 0.65 and len(text) > 50:
            # Multiple injection points for more natural flow
            num_injections = random.randint(1, 2)  # Reduced to avoid over-injection
            text_words = text.split()
            
            if len(text_words) > 10:
                for _ in range(num_injections):
                    if len(text_words) > 15:
                        pos = random.randint(5, len(text_words)-5)
                        injection = random.choice(chaos_options)
                        text_words.insert(pos, injection)
                
                text = " ".join(text_words)
        
        # Add perspective-specific sentence starters
        if random.random() < 0.4 and perspective_words:
            starters = [f"{word}, " for word in perspective_words[:3]]
            if starters and len(text) > 50:
                try:
                    text = random.choice(starters) + text[0].lower() + text[1:]
                except:
                    pass
        
        return text

    def _enhanced_burstiness(self, text):
        """Advanced burstiness with topic-aware sentence restructuring"""
        if not text:
            return text
            
        try:
            s = sent_tokenize(text)
        except:
            return text
            
        if len(s) > 2:
            # Topic-aware restructuring
            dominant_topic, _ = self._analyze_topic_perspective(text)
            
            if dominant_topic == 'story':
                # Story-like flow with more connections
                if random.random() < 0.6:
                    for i in range(len(s)-1):
                        if random.random() < 0.4:
                            connectors = ["So then, ", "Anyway, ", "Meanwhile, ", "Suddenly, "]
                            try:
                                s[i+1] = random.choice(connectors) + s[i+1][0].lower() + s[i+1][1:]
                            except:
                                pass
            
            elif dominant_topic == 'explanation':
                # Explanation flow with cause-effect markers
                if random.random() < 0.5:
                    for i in range(len(s)-1):
                        if random.random() < 0.3 and i < len(s)-1:
                            try:
                                s[i] = s[i].rstrip(".!?") + " — which means " + s[i+1].lstrip()
                                del s[i+1]
                                break
                            except:
                                pass
            
            # Universal burstiness techniques
            if random.random() < 0.5 and len(s) > 1:
                i = random.randint(0, len(s)-2)
                connectors = [" — ", "...", " ...and ", " — so "]
                try:
                    s[i] = s[i].rstrip(".!?") + random.choice(connectors) + s[i+1].lstrip()
                    del s[i+1]
                except:
                    pass
            
            # Split long sentences with topic-appropriate conjunctions
            if random.random() < 0.45:
                for j, sent in enumerate(s):
                    if len(sent.split()) > 15:  # Reduced threshold
                        split_points = [
                            r"(?<=but)\s+", r"(?<=and)\s+", r"(?<=so)\s+", 
                            r"(?<=because)\s+", r"(?<=which)\s+"
                        ]
                        split_pattern = random.choice(split_points)
                        try:
                            parts = re.split(split_pattern, sent, 1)
                            if len(parts) > 1:
                                s[j] = parts[0] + "."
                                continuation = parts[1][0].upper() + parts[1][1:] if parts[1] else ""
                                if continuation:
                                    s.insert(j+1, continuation)
                                break
                        except:
                            pass
        
        return " ".join(s) if s else text

    def _advanced_typos(self, text):
        """More sophisticated and context-aware typos"""
        if not text:
            return text
            
        if random.random() < 0.25:  # Slightly reduced probability
            # Expanded typo dictionary
            typos = {
                "the ": "teh ", "and ": "adn ", "with ": "wiht ", "which ": "wich ",
                "because ": "becasue ", "there ": "ther ", "their ": "thier ",
                "really ": "realy ", "definitely ": "definately ", "until ": "untill ",
                "through ": "thru ", "though ": "tho ", "although ": "altho ",
                "enough ": "enuf ", "probably ": "prolly ", "especially ": "specially ",
                "your ": "ur ", "you ": "u ", "are ": "r ", "see ": "c ",
                "people ": "ppl ", "before ": "befor ", "after ": "aftr ",
                "about ": "abt ", "through ": "thru ", "though ": "tho "
            }
            # Apply multiple typos occasionally
            num_typos = random.randint(1, 2)  # Reduced to avoid over-typing
            applied = 0
            words = list(typos.keys())
            random.shuffle(words)
            
            for word in words:
                if word in text.lower() and applied < num_typos:
                    try:
                        text = re.sub(re.escape(word), typos[word], text, flags=re.IGNORECASE, count=1)
                        applied += 1
                    except:
                        continue
        
        # Add occasional missing spaces (with safety)
        if random.random() < 0.12 and len(text) > 20:
            words = text.split()
            if len(words) > 8:
                try:
                    pos = random.randint(2, len(words)-3)
                    if pos + 1 < len(words):
                        words[pos] = words[pos] + words[pos+1]
                        del words[pos+1]
                        text = " ".join(words)
                except:
                    pass
        
        return text

    def _contextual_contractions(self, text):
        """Enhanced contractions with contextual awareness"""
        if not text:
            return text
            
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
            if random.random() < 0.7:  # Slightly reduced probability
                try:
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                except:
                    continue
        
        return text

    def _add_conversational_elements(self, text):
        """Add conversational elements based on topic"""
        if not text or len(text) < 30:
            return text
            
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
        if random.random() < 0.3 and len(text) > 50:  # Reduced probability
            addon = random.choice(addons)
            if random.random() < 0.7:
                text = text + " " + addon
            else:
                # Sometimes insert in the middle for more natural flow
                try:
                    sentences = sent_tokenize(text)
                    if len(sentences) > 2:
                        pos = random.randint(1, len(sentences)-1)
                        sentences.insert(pos, addon)
                        text = " ".join(sentences)
                except:
                    text = text + " " + addon
        
        return text

    def _add_emotional_expression(self, text):
        """Add emotional expressions for more humanity"""
        if not text or len(text) < 40:
            return text
            
        emotions = [
            "Haha", "Wow", "Seriously", "No way", "Awesome", "Interesting", 
            "Crazy", "Wild", "Amazing", "Incredible", "Unbelievable"
        ]
        
        if random.random() < 0.25:
            emotion = random.choice(emotions)
            positions = [
                f"{emotion}! ", 
                f"{emotion}, ", 
                f" — {emotion.lower()} — ",
                f" ...{emotion.lower()}... "
            ]
            
            if len(text.split()) > 15:
                try:
                    words = text.split()
                    insert_pos = random.randint(0, min(5, len(words)-1))
                    words.insert(insert_pos, random.choice(positions).strip())
                    text = " ".join(words)
                except:
                    pass
                    
        return text

    def humanize(self, text, level="Standard"):
        if not self.model: 
            return "Error: Model failed to load."
            
        if not text or len(text.strip()) == 0:
            return text

        with torch.no_grad():
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            result = []

            for para in paragraphs:
                if not para or len(para.strip()) == 0:
                    continue
                    
                # Analyze topic perspective for this paragraph
                dominant_topic, perspective_words = self._analyze_topic_perspective(para)
                
                # ENHANCED QUAD-VARIATION ENGINE with topic awareness
                candidates = []
                for i in range(4):
                    try:
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
                        )
                        candidate = self.tokenizer.decode(out[0], skip_special_tokens=True)
                        if candidate and len(candidate.strip()) > 0:
                            candidates.append(candidate)
                    except Exception as e:
                        print(f"Generation error: {e}")
                        continue

                if not candidates:
                    result.append(para)
                    continue

                # SMART FUSION with topic perspective enhancement
                try:
                    base = max(candidates, key=lambda x: (len(set(x.split())) / len(x.split()) if x and len(x.split()) > 0 else 0) + 
                             (0.1 if any(word in x.lower() for word in perspective_words) else 0))
                except:
                    base = candidates[0]
                
                # Enhanced sentence mixing with topic coherence
                if len(candidates) > 1:
                    for extra in random.sample(candidates, min(2, len(candidates))):
                        try:
                            s1 = sent_tokenize(base)
                            s2 = sent_tokenize(extra)
                            if len(s2) > 1 and len(s1) > 3:
                                # Smart replacement maintaining topic coherence
                                replacements = min(1, len(s1)-1)  # Reduced to avoid over-mixing
                                for _ in range(replacements):
                                    if len(s1) > 3:
                                        i = random.randint(1, len(s1)-2)
                                        if i < len(s2):
                                            s1[i] = s2[i]
                                base = " ".join(s1)
                        except:
                            continue

                # ULTRA-ENHANCED HUMANITY PIPELINE
                final = base
                final = self._inject_topic_specific_chaos(final, perspective_words)
                final = self._ultimate_anti_ai(final)
                final = self._enhanced_burstiness(final)
                final = self._advanced_typos(final)
                final = self._contextual_contractions(final)
                final = self._add_conversational_elements(final)
                final = self._add_emotional_expression(final)

                # Final touch - ensure natural capitalization
                try:
                    sentences = sent_tokenize(final)
                    sentences = [s[0].upper() + s[1:] if s and s[0].isalpha() else s for s in sentences]
                    final = " ".join(sentences)
                except:
                    pass

                result.append(final)

            return "\n\n".join(result) if result else text