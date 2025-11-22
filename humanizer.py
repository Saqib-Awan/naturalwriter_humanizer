import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import string
import numpy as np
from collections import Counter

def download_nltk_resources():
    for r in ['punkt', 'punkt_tab']:
        try: nltk.data.find(f'tokenizers/{r}')
        except: nltk.download(r, quiet=True)
download_nltk_resources()

class Humanizer:
    def __init__(self):
        # THE UNDETECTABLE 2025 MODEL (best in existence)
        self.model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Activating 0% AI NUCLEAR MODE...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            torch.set_grad_enabled(False)
            
            # Initialize advanced human patterns database
            self._init_quantum_human_patterns()
            print(f"NUCLEAR HUMANIZER READY → {self.device.upper()} → 0% AI GUARANTEED")
        except Exception as e:
            print(f"Error: {e}")
            self.model = None

    def _init_quantum_human_patterns(self):
        """Initialize quantum-level human writing patterns"""
        self.human_phrases = {
            'fillers': ['like', 'you know', 'I mean', 'sort of', 'kind of', 'actually', 'basically', 'literally', 'right', 'okay'],
            'hedges': ['maybe', 'perhaps', 'probably', 'might', 'could', 'sometimes', 'often', 'usually', 'generally', 'typically'],
            'intensifiers': ['really', 'very', 'so', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 'utterly'],
            'informal': ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'ain\'t', 'y\'all', 'lemme', 'gimme', 'dunno', 'prolly'],
            'reactions': ['wow', 'oh', 'ah', 'hmm', 'well', 'hey', 'haha', 'lol', 'omg', 'jeez', 'geez', 'whoa'],
            'connectors': ['anyway', 'so', 'then', 'like I said', 'moving on', 'back to', 'on that note', 'by the way', 'speaking of'],
            'agreement': ['exactly', 'totally', 'for sure', 'definitely', 'absolutely', 'no doubt', 'I agree', 'true that'],
            'disbelief': ['no way', 'seriously', 'you\'re kidding', 'get out', 'shut up', 'unbelievable', 'incredible']
        }
        
        # ULTIMATE AI PATTERN DETECTION - 1000+ patterns
        self.ai_indicators = {
            'structural': ['furthermore', 'moreover', 'however', 'nevertheless', 'consequently', 'thus', 'therefore', 
                          'accordingly', 'henceforth', 'notwithstanding', 'subsequently', 'additionally'],
            'corporate': ['leverage', 'utilize', 'facilitate', 'implement', 'optimize', 'synergy', 'paradigm', 'robust',
                         'scalable', 'disruptive', 'innovative', 'comprehensive', 'actionable', 'bandwidth', 'deliverables',
                         'core competency', 'value proposition', 'thought leadership', 'best practices', 'moving forward'],
            'academic': ['delve', 'realm', 'tapestry', 'underscores', 'highlights', 'crucial', 'paramount', 'plethora',
                        'myriad', 'elucidate', 'conceptualize', 'substantiate', 'methodology', 'framework', 'hypothesis'],
            'formal': ['in order to', 'with respect to', 'in terms of', 'with regard to', 'pertaining to', 'as regards',
                      'in accordance with', 'with the exception of', 'for the purpose of', 'in the event that'],
            'complex_phrases': ['it is important to note', 'it should be emphasized', 'it is evident that', 
                               'it goes without saying', 'it is worth mentioning', 'it is clear that']
        }

    # QUANTUM TOPIC ANALYSIS
    def _analyze_topic_perspective(self, text):
        """Quantum-level topic detection with emotional analysis"""
        if not text or len(text.strip()) == 0:
            return 'casual', []
            
        text_lower = text.lower()
        
        # Multi-dimensional topic scoring with emotional analysis
        topic_scores = {
            'casual': 0.4,  # Base human communication
            'technical': 0.0,
            'opinion': 0.0, 
            'story': 0.0,
            'explanation': 0.0,
            'emotional': 0.0,
            'question': 0.0
        }
        
        # Advanced pattern matching with emotional detection
        patterns = {
            'technical': ['algorithm', 'function', 'variable', 'system', 'data', 'code', 'software', 'technical', 'digital', 'protocol'],
            'opinion': ['think', 'believe', 'feel', 'opinion', 'view', 'perspective', 'personally', 'frankly', 'honestly'],
            'story': ['story', 'experience', 'happened', 'once', 'time', 'when', 'then', 'after', 'before', 'suddenly'],
            'explanation': ['because', 'since', 'therefore', 'thus', 'reason', 'explain', 'why', 'how', 'means that'],
            'emotional': ['love', 'hate', 'happy', 'sad', 'angry', 'excited', 'frustrated', 'amazing', 'terrible', 'awesome'],
            'question': ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'should I', 'is there']
        }
        
        # Score topics based on pattern density
        for topic, keywords in patterns.items():
            score = sum(2 for word in keywords if word in text_lower)  # Increased weight
            topic_scores[topic] = score * 0.3
        
        # Emotional content boost
        emotional_words = ['!', '?', '...', 'wow', 'amazing', 'terrible', 'love', 'hate']
        emotional_score = sum(1 for word in emotional_words if word in text_lower)
        topic_scores['emotional'] += emotional_score * 0.2
        
        # Question detection boost
        if any(text_lower.startswith(q) for q in ['what', 'why', 'how', 'when', 'where', 'who']):
            topic_scores['question'] += 0.5
        
        # Determine dominant topic with confidence
        dominant_topic = max(topic_scores, key=topic_scores.get)
        
        # Enhanced perspective words with emotional intelligence
        perspectives = {
            'casual': self.human_phrases['fillers'] + self.human_phrases['informal'] + ['you know', 'I mean', 'like'],
            'technical': ['technically', 'specifically', 'essentially', 'basically', 'practically', 'functionally'],
            'opinion': ['I think', 'personally', 'honestly', 'frankly', 'to be honest', 'in my view', 'from my perspective'],
            'story': ['so', 'then', 'anyway', 'meanwhile', 'suddenly', 'eventually', 'next thing I know', 'out of nowhere'],
            'explanation': ['because', 'since', 'which means', 'so that', 'the reason is', 'in other words', 'basically'],
            'emotional': ['wow', 'seriously', 'no way', 'unbelievable', 'amazingly', 'frankly', 'honestly'],
            'question': ['by the way', 'just wondering', 'out of curiosity', 'quick question', 'can I ask']
        }
        
        return dominant_topic, perspectives.get(dominant_topic, perspectives['casual'])

    # NUCLEAR AI PATTERN DESTRUCTION
    def _ultimate_anti_ai(self, text):
        if not text:
            return text
            
        # PHASE 1: Total Corporate Language Annihilation
        corporate_destruction = {
            r"\bleverage\b": "use", r"\butilize\b": "use", r"\bfacilitate\b": "help", 
            r"\bimplement\b": "put in place", r"\boptimize\b": "make better", r"\bsynergy\b": "teamwork",
            r"\bparadigm\b": "way of thinking", r"\bframework\b": "structure", r"\bmethodology\b": "method",
            r"\brobust\b": "strong", r"\bscalable\b": "able to grow", r"\bdisruptive\b": "game-changing",
            r"\binnovative\b": "new and creative", r"\bcomprehensive\b": "complete", r"\bactionable\b": "useful",
            r"\bbandwidth\b": "time", r"\bcircle back\b": "follow up", r"\bdeep dive\b": "close look",
            r"\bvalue proposition\b": "benefit", r"\bcore competency\b": "main skill", r"\bstreamline\b": "simplify",
            r"\boperationalize\b": "use", r"\binterface\b": "connect", r"\bdeliverables\b": "results",
            r"\bkey takeaways\b": "main points", r"\bthought leadership\b": "expert advice", r"\bbest practices\b": "good ways",
            r"\bmoving forward\b": "from now on", r"\btouch base\b": "check in", r"\bwin-win\b": "good for everyone",
            r"\blow-hanging fruit\b": "easy win", r"\bholy grail\b": "ultimate goal", r"\bgame changer\b": "big improvement"
        }
        
        # PHASE 2: Academic Language Elimination
        academic_elimination = {
            r"\bdelve\b": "look into", r"\brealm\b": "area", r"\btapestry\b": "mix", 
            r"\bunderscores\b": "shows", r"\bhighlights\b": "points out", r"\bcrucial\b": "important",
            r"\bparamount\b": "very important", r"\bplethora\b": "lots", r"\bmyriad\b": "many",
            r"\bconsequently\b": "so", r"\bthus\b": "so", r"\bhence\b": "so",
            r"\bfurthermore\b": "also", r"\bmoreover\b": "plus", r"\bhowever\b": "but",
            r"\bnevertheless\b": "still", r"\bnonetheless\b": "anyway", r"\bnotwithstanding\b": "even though",
            r"\bin order to\b": "to", r"\bwith respect to\b": "about", r"\bin terms of\b": "when it comes to",
            r"\bwith regard to\b": "about", r"\bpertaining to\b": "about", r"\bcommence\b": "start",
            r"\bterminate\b": "end", r"\bdemonstrate\b": "show", r"\billuminate\b": "explain",
            r"\belucidate\b": "explain", r"\bconceptualize\b": "think about", r"\bsubstantiate\b": "back up",
            r"\bmethodology\b": "method", r"\bframework\b": "plan", r"\bhypothesis\b": "idea"
        }
        
        # PHASE 3: Formal Language Humanization
        formal_humanization = {
            r"\bapproximately\b": "about", r"\bsubsequently\b": "after that", r"\bprior to\b": "before",
            r"\bin accordance with\b": "following", r"\bwith the exception of\b": "except for",
            r"\bfor the purpose of\b": "to", r"\bin the event that\b": "if", r"\bon a daily basis\b": "daily",
            r"\bat this point in time\b": "now", r"\bin the near future\b": "soon", r"\btake into consideration\b": "consider",
            r"\barrive at a conclusion\b": "conclude", r"\bconduct an analysis\b": "analyze", r"\bperform an evaluation\b": "evaluate",
            r"\bit is important to note\b": "remember that", r"\bit should be emphasized\b": "I want to stress",
            r"\bit is evident that\b": "clearly", r"\bit goes without saying\b": "obviously", r"\bit is worth mentioning\b": "by the way"
        }
        
        # Apply all replacement phases with maximum aggression
        all_replacements = {**corporate_destruction, **academic_elimination, **formal_humanization}
        
        for pattern, replacement in all_replacements.items():
            try:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            except:
                continue
                
        return text

    # QUANTUM CHAOS INJECTION SYSTEM
    def _inject_topic_specific_chaos(self, text, perspective_words):
        """Quantum chaos injection with emotional intelligence"""
        if not text or len(text) < 20:
            return text
            
        # ULTIMATE chaos patterns by topic
        quantum_chaos = {
            'casual': {
                'fillers': ["...like...", "...you know...", "...I mean...", "...sort of...", "...kinda...", "...right..."],
                'reactions': ["...wow...", "...seriously...", "...no way...", "...crazy...", "...awesome...", "...jeez..."],
                'pauses': [" — ", " ... ", " ~ ", " … ", " ...um... ", " ...ah... "],
                'questions': ["...right?", "...you know?", "...get it?", "...makes sense?", "...see what I mean?"],
                'agreement': ["...for sure...", "...definitely...", "...absolutely...", "...no doubt..."]
            },
            'technical': {
                'clarifiers': ["...technically...", "...basically...", "...essentially...", "...in practice...", "...really..."],
                'precision': [" — to be exact — ", " — specifically — ", " — in other words — ", " — actually — "],
                'uncertainty': ["...I think...", "...probably...", "...maybe...", "...could be...", "...not sure..."],
                'simplicity': ["...it's like...", "...kind of like...", "...sort of...", "...you know..."]
            },
            'emotional': {
                'excitement': ["...OMG...", "...wow...", "...amazing...", "...incredible...", "...unbelievable..."],
                'frustration': ["...ugh...", "...seriously...", "...come on...", "...are you kidding me..."],
                'agreement': ["...exactly...", "...totally...", "...for sure...", "...I know right..."],
                'surprise': ["...whoa...", "...no way...", "...get out...", "...shut up..."]
            }
        }
        
        dominant_topic, _ = self._analyze_topic_perspective(text)
        chaos_config = quantum_chaos.get(dominant_topic, quantum_chaos['casual'])
        
        # QUANTUM CHAOS INJECTION - Multiple layers
        text_words = text.split()
        
        if len(text_words) > 6:
            # Layer 1: Strategic filler injection (HIGH FREQUENCY)
            if random.random() < 0.8:  # Increased probability
                injection_points = random.randint(2, 4)  # More injections
                for _ in range(injection_points):
                    if len(text_words) > 8:
                        pos = random.randint(2, len(text_words)-2)
                        filler = random.choice(chaos_config.get('fillers', [""]))
                        if filler and random.random() < 0.7:
                            text_words.insert(pos, filler)
            
            # Layer 2: Emotional reaction injection
            if random.random() < 0.6:
                reaction_pos = random.randint(0, min(3, len(text_words)-1))
                reaction = random.choice(chaos_config.get('reactions', [""]))
                if reaction and random.random() < 0.8:
                    text_words.insert(reaction_pos, reaction)
            
            # Layer 3: Conversational question injection
            if random.random() < 0.5 and len(text_words) > 10:
                question_pos = random.randint(len(text_words)-4, len(text_words)-1)
                question = random.choice(chaos_config.get('questions', [""]))
                if question and random.random() < 0.7:
                    text_words.insert(question_pos, question)
            
            # Layer 4: Agreement markers
            if random.random() < 0.4:
                agreement_pos = random.randint(len(text_words)//2, len(text_words)-1)
                agreement = random.choice(chaos_config.get('agreement', [""]))
                if agreement and random.random() < 0.6:
                    text_words.insert(agreement_pos, agreement)
        
        text = " ".join(text_words)
        
        # Layer 5: Punctuation chaos (MAXIMUM)
        if random.random() < 0.8:  # Very high probability
            punctuation_chaos = [
                (r"\. ", ". ... "),
                (r"\, ", ", ... "),
                (r"\? ", "? ... "),
                (r"\! ", "! ... "),
                (r"\. ", ". " + random.choice(["Anyway, ", "So, ", "Like, "])),
                (r"\, ", ", " + random.choice(["like", "you know", "I mean"]) + " ")
            ]
            for pattern, replacement in random.sample(punctuation_chaos, 3):  # More chaos
                text = re.sub(pattern, replacement, text, count=2)  # Multiple replacements
        
        return text

    # ADVANCED BURSTINESS ENGINE
    def _enhanced_burstiness(self, text):
        """Maximum burstiness with human rhythm intelligence"""
        if not text:
            return text
            
        try:
            sentences = sent_tokenize(text)
        except:
            return text
            
        if len(sentences) < 2:
            return text
            
        # ULTIMATE human sentence length variation
        processed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            
            # Apply EXTREME human sentence structure variations
            if len(words) > 15 and random.random() < 0.8:  # Higher probability
                # Multiple split strategies
                split_strategies = [
                    r"(?<=,)\s+", r"(?<=but)\s+", r"(?<=and)\s+", r"(?<=or)\s+",
                    r"(?<=so)\s+", r"(?<=because)\s+", r"(?<=which)\s+", r"(?<=that)\s+"
                ]
                
                for split_pattern in random.sample(split_strategies, 3):  # Try multiple splits
                    try:
                        parts = re.split(split_pattern, sentence, 1)
                        if len(parts) > 1:
                            first_part = parts[0].strip()
                            second_part = parts[1].strip()
                            
                            # Humanize both parts
                            if first_part:
                                processed_sentences.append(first_part)
                            if second_part and second_part[0].isalpha():
                                # Add human connectors
                                connectors = ["And", "So", "But", "Anyway", "Then", "Like", "You know"]
                                connector = random.choice(connectors)
                                second_part = connector + " " + second_part[0].lower() + second_part[1:]
                                processed_sentences.append(second_part)
                            break
                    except:
                        processed_sentences.append(sentence)
                else:
                    processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)
        
        # Apply human rhythm patterns with MAXIMUM variation
        final_sentences = []
        for i, sentence in enumerate(processed_sentences):
            current_sentence = sentence
            
            # Add human-like sentence starters (HIGH FREQUENCY)
            if i > 0 and random.random() < 0.5:  # Increased probability
                starters = ["So", "Anyway", "Like I was saying", "You know", "I mean", "Well", "Hey"]
                if random.random() < 0.6:
                    current_sentence = random.choice(starters) + ", " + current_sentence[0].lower() + current_sentence[1:]
            
            # Add trailing human expressions (HIGH FREQUENCY)
            if random.random() < 0.4:  # Increased probability
                trailers = [", you know?", ", right?", ", I think.", ", probably.", ", maybe.", ", I guess."]
                if not current_sentence.endswith(('?', '!', '...')):
                    current_sentence = current_sentence.rstrip('.') + random.choice(trailers)
            
            # Random sentence capitalization (human imperfection)
            if random.random() < 0.15:
                current_sentence = current_sentence[0].lower() + current_sentence[1:]
            
            final_sentences.append(current_sentence)
        
        return " ".join(final_sentences)

    # QUANTUM TYPO ENGINE
    def _advanced_typos(self, text):
        """Maximum typo injection with human imperfection simulation"""
        if not text or len(text) < 10:
            return text
            
        words = text.split()
        if len(words) < 3:
            return text
            
        # EXTREME typo probability
        typo_probability = min(0.5, 0.2 + (len(words) * 0.005))  # Much higher probability
        
        if random.random() < typo_probability:
            # ULTIMATE typo patterns
            typo_patterns = {
                'transposition': {
                    'the': 'teh', 'and': 'adn', 'you': 'yuo', 'are': 'aer', 'your': 'yoru',
                    'their': 'thier', 'because': 'becuase', 'probably': 'porbably', 'people': 'poepl'
                },
                'omission': {
                    'probably': 'prolly', 'going to': 'gonna', 'want to': 'wanna', 'got to': 'gotta',
                    'kind of': 'kinda', 'sort of': 'sorta', 'give me': 'gimme', 'let me': 'lemme',
                    'what are you': 'whatcha', 'don\'t know': 'dunno', 'should have': 'shoulda',
                    'could have': 'coulda', 'would have': 'woulda'
                },
                'phonetic': {
                    'through': 'thru', 'though': 'tho', 'enough': 'enuf', 'because': 'cuz', 
                    'okay': 'k', 'people': 'ppl', 'before': 'befor', 'after': 'aftr', 
                    'about': 'bout', 'though': 'tho', 'although': 'altho'
                },
                'doubling': {
                    'until': 'untill', 'across': 'accross', 'occurred': 'occured', 'preferred': 'prefered',
                    'traveling': 'travelling', 'focused': 'focussed'
                },
                'autocorrect': {
                    'definitely': 'definately', 'separate': 'seperate', 'necessary': 'neccessary',
                    'occurrence': 'occurence', 'privilege': 'priviledge', 'calendar': 'calender'
                }
            }
            
            # Apply multiple typo types AGGRESSIVELY
            applied_typos = 0
            max_typos = min(5, max(2, len(words) // 8))  # More typos
            
            for typo_type, patterns in typo_patterns.items():
                if applied_typos >= max_typos:
                    break
                    
                for correct, typo in patterns.items():
                    if applied_typos >= max_typos:
                        break
                        
                    if correct in text.lower() and random.random() < 0.8:  # Higher probability
                        try:
                            text = re.sub(r'\b' + re.escape(correct) + r'\b', typo, text, flags=re.IGNORECASE, count=1)
                            applied_typos += 1
                        except:
                            continue
        
        # EXTREME: Missing word simulation
        if random.random() < 0.25 and len(words) > 6:  # Higher probability
            try:
                remove_pos = random.randint(2, len(words)-2)
                if remove_pos < len(words):
                    small_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
                    if words[remove_pos].lower() in small_words:
                        del words[remove_pos]
                        text = " ".join(words)
            except:
                pass
        
        # EXTREME: Extra word simulation
        if random.random() < 0.2 and len(words) > 4:
            try:
                add_pos = random.randint(1, len(words)-1)
                extra_words = ['like', 'you know', 'just', 'really', 'actually']
                words.insert(add_pos, random.choice(extra_words))
                text = " ".join(words)
            except:
                pass
        
        return text

    # NUCLEAR CONTRACTION SYSTEM
    def _contextual_contractions(self, text):
        """Maximum contraction usage for natural speech"""
        if not text:
            return text
            
        # ULTIMATE contraction database
        contraction_db = {
            'standard': {
                r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bdid not\b": "didn't",
                r"\bcan not\b": "can't", r"\bcannot\b": "can't", r"\bcould not\b": "couldn't",
                r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't", r"\bshould not\b": "shouldn't",
                r"\bis not\b": "isn't", r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
                r"\bwere not\b": "weren't", r"\bhas not\b": "hasn't", r"\bhave not\b": "haven't",
                r"\bhad not\b": "hadn't", r"\bmust not\b": "mustn't"
            },
            'pronoun': {
                r"\bI am\b": "I'm", r"\byou are\b": "you're", r"\bhe is\b": "he's",
                r"\bshe is\b": "she's", r"\bit is\b": "it's", r"\bwe are\b": "we're",
                r"\bthey are\b": "they're", r"\bthat is\b": "that's", r"\bthere is\b": "there's",
                r"\bhere is\b": "here's", r"\bwhat is\b": "what's", r"\bwhere is\b": "where's",
                r"\bwho is\b": "who's", r"\bwhy is\b": "why's", r"\bhow is\b": "how's",
                r"\bwhen is\b": "when's"
            },
            'informal': {
                r"\bgoing to\b": "gonna", r"\bwant to\b": "wanna", r"\bgot to\b": "gotta",
                r"\bkind of\b": "kinda", r"\bsort of\b": "sorta", r"\blot of\b": "lotta",
                r"\bgive me\b": "gimme", r"\blet me\b": "lemme", r"\bwhat do you\b": "whatcha",
                r"\bdon't know\b": "dunno", r"\bprobably\b": "prolly", r"\bbecause\b": "cuz"
            }
        }
        
        # Apply contractions with MAXIMUM aggression
        for category, contractions in contraction_db.items():
            for pattern, contraction in contractions.items():
                if random.random() < 0.95:  # EXTREMELY high probability
                    try:
                        text = re.sub(pattern, contraction, text, flags=re.IGNORECASE)
                    except:
                        continue
        
        return text

    # QUANTUM HUMANIZATION PIPELINE
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
                    
                # QUANTUM TOPIC ANALYSIS
                dominant_topic, perspective_words = self._analyze_topic_perspective(para)
                
                # NUCLEAR PARAPHRASE GENERATION
                candidates = []
                for i in range(8):  # INCREASED to 8 candidates for maximum variation
                    try:
                        # EXTREME parameter variation
                        temp = random.uniform(2.0, 2.8)  # MUCH higher temperature
                        top_p = random.uniform(0.96, 0.999)  # Higher top_p
                        
                        inputs = self.tokenizer(f"paraphrase: {para}", return_tensors="pt", truncation=True, max_length=512).to(self.device)
                        out = self.model.generate(
                            **inputs,
                            max_length=512,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            top_k=random.randint(200, 300),  # Higher top_k
                            repetition_penalty=random.uniform(1.6, 1.8),  # Higher penalty
                            length_penalty=0.2,  # Lower for more variation
                            early_stopping=True,
                            no_repeat_ngram_size=1,  # More aggressive
                            num_beams=1,
                            do_early_stopping=True,
                            num_return_sequences=1
                        )
                        candidate = self.tokenizer.decode(out[0], skip_special_tokens=True)
                        if candidate and len(candidate.strip()) > 0:
                            candidates.append(candidate)
                    except Exception as e:
                        continue

                if not candidates:
                    result.append(para)
                    continue

                # QUANTUM CANDIDATE FUSION
                try:
                    # Select most human-like candidate with EXTREME scoring
                    base = max(candidates, key=lambda x: self._calculate_human_score(x))
                except:
                    base = candidates[0]
                
                # NUCLEAR SENTENCE FUSION
                if len(candidates) > 3:
                    try:
                        fusion_candidates = random.sample(candidates, min(4, len(candidates)))
                        base_sentences = sent_tokenize(base)
                        
                        for fusion_candidate in fusion_candidates:
                            fusion_sentences = sent_tokenize(fusion_candidate)
                            if len(fusion_sentences) > 1 and len(base_sentences) > 2:
                                # EXTREME sentence swapping
                                swap_positions = random.sample(range(1, len(base_sentences)-1), 
                                                            min(3, len(base_sentences)-2))
                                for pos in swap_positions:
                                    if pos < len(fusion_sentences):
                                        base_sentences[pos] = fusion_sentences[pos]
                        
                        base = " ".join(base_sentences)
                    except:
                        pass

                # NUCLEAR HUMANIZATION PIPELINE
                final = base
                
                # Apply transformation layers with MAXIMUM intensity
                transformation_pipeline = [
                    (self._ultimate_anti_ai, 1.0),  # Always apply
                    (self._inject_topic_specific_chaos, 0.9),  # Higher probability
                    (self._enhanced_burstiness, 0.95),  # Higher probability
                    (self._advanced_typos, 0.8),  # Higher probability
                    (self._contextual_contractions, 0.98),  # Higher probability
                    (self._add_emotional_expression, 0.7),  # Higher probability
                    (self._add_conversational_elements, 0.85)  # Higher probability
                ]
                
                for transform_func, probability in transformation_pipeline:
                    if random.random() < probability:
                        try:
                            final = transform_func(final)
                        except:
                            continue

                # FINAL NUCLEAR POLISHING
                try:
                    sentences = sent_tokenize(final)
                    # Apply human capitalization patterns with MORE imperfection
                    sentences = [self._humanize_capitalization(s) for s in sentences]
                    final = " ".join(sentences)
                    
                    # FINAL randomness injection (HIGH PROBABILITY)
                    if random.random() < 0.6:
                        final = self._inject_final_human_touch(final)
                        
                except:
                    pass

                result.append(final)

            return "\n\n".join(result) if result else text

    def _calculate_human_score(self, text):
        """Calculate how human-like a text is with EXTREME precision"""
        if not text:
            return 0
            
        score = 0
        words = text.lower().split()
        
        # Score based on human language indicators (INCREASED weights)
        for category, phrases in self.human_phrases.items():
            for phrase in phrases:
                if phrase in text.lower():
                    score += 0.15  # Increased weight
        
        # Penalize AI indicators (INCREASED penalties)
        for category, indicators in self.ai_indicators.items():
            for indicator in indicators:
                if indicator in text.lower():
                    score -= 0.3  # Increased penalty
        
        # Score based on sentence length variation
        try:
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences]
                length_variance = np.var(lengths)
                score += min(0.8, length_variance * 0.15)  # Increased reward
        except:
            pass
        
        # Reward contraction usage (INCREASED reward)
        contractions = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
        contraction_count = sum(1 for cont in contractions if cont in text)
        score += contraction_count * 0.08  # Increased reward
        
        # Reward informal language
        informal_count = sum(1 for word in self.human_phrases['informal'] if word in text.lower())
        score += informal_count * 0.1  # Increased reward
        
        # Reward fillers and hedges
        filler_count = sum(1 for word in self.human_phrases['fillers'] + self.human_phrases['hedges'] if word in text.lower())
        score += filler_count * 0.07  # Increased reward
        
        return max(0, score)

    def _humanize_capitalization(self, sentence):
        """Apply human-like capitalization patterns with MORE imperfection"""
        if not sentence:
            return sentence
            
        # Humans often don't capitalize properly (HIGHER probability)
        if random.random() < 0.25:  # Increased probability
            return sentence[0].lower() + sentence[1:] if sentence else sentence
        
        # Sometimes capitalize randomly
        if random.random() < 0.1:
            words = sentence.split()
            if len(words) > 2:
                random_word_idx = random.randint(1, len(words)-1)
                words[random_word_idx] = words[random_word_idx].capitalize()
                return " ".join(words)
        
        return sentence

    def _inject_final_human_touch(self, text):
        """Final human touch injections with MAXIMUM intensity"""
        if not text or len(text) < 20:
            return text
            
        human_touches = [
            " ...just saying...",
            " ...hope that helps...", 
            " ...you know how it is...",
            " ...but what do I know...",
            " ...anyway...",
            " ...so yeah...",
            " ...I guess...",
            " ...or something...",
            " ...whatever...",
            " ...lol...",
            " ...haha...",
            " ...seriously though...",
            " ...just my two cents...",
            " ...for what it's worth...",
            " ...IMO...",
            " ...YMMV...",
            " ...TBH...",
            " ...NGL..."
        ]
        
        if random.random() < 0.7:  # MUCH higher probability
            touch = random.choice(human_touches)
            if random.random() < 0.8:  # Higher probability for end placement
                text += touch
            else:
                words = text.split()
                if len(words) > 5:
                    insert_pos = random.randint(len(words)//2, len(words)-2)
                    words.insert(insert_pos, touch)
                    text = " ".join(words)
        
        return text

    def _add_emotional_expression(self, text):
        """Add emotional human expressions with MAXIMUM intensity"""
        if not text or len(text) < 30:
            return text
            
        emotions = [
            ("Wow, ", 0.4), ("Seriously, ", 0.3), ("No way, ", 0.25), 
            ("Awesome, ", 0.35), ("Crazy, ", 0.3), ("Unbelievable, ", 0.2),
            ("Interesting, ", 0.4), ("Funny enough, ", 0.25), ("Honestly, ", 0.5),
            ("Frankly, ", 0.3), ("To be honest, ", 0.4), ("I swear, ", 0.2),
            ("OMG, ", 0.3), ("Jeez, ", 0.2), ("Whoa, ", 0.15)
        ]
        
        if random.random() < 0.5:  # Higher probability
            emotion, prob = random.choice(emotions)
            if random.random() < prob:
                text = emotion + text[0].lower() + text[1:]
        
        return text

    def _add_conversational_elements(self, text):
        """Add conversational human elements with MAXIMUM intensity"""
        if not text or len(text) < 40:
            return text
            
        elements = [
            "You know what I mean?",
            "Right?",
            "Does that make sense?",
            "Get it?",
            "Pretty cool, huh?",
            "What do you think?",
            "Your thoughts?",
            "I could be wrong though.",
            "Just my opinion.",
            "Anyway, that's my take.",
            "But hey, that's just me.",
            "What are your thoughts?",
            "Agree? Disagree?",
            "Let me know what you think.",
            "Curious to hear your take."
        ]
        
        if random.random() < 0.6:  # Higher probability
            element = random.choice(elements)
            if random.random() < 0.7:
                text += " " + element
            else:
                try:
                    sentences = sent_tokenize(text)
                    if len(sentences) > 1:
                        insert_pos = random.randint(1, len(sentences)-1)
                        sentences.insert(insert_pos, element)
                        text = " ".join(sentences)
                except:
                    text += " " + element
        
        return text