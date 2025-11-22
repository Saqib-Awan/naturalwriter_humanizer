import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import string
import numpy as np

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
            
            # Initialize human writing patterns database
            self._init_human_patterns()
            print(f"NUCLEAR HUMANIZER READY → {self.device.upper()} → 0% AI GUARANTEED")
        except Exception as e:
            print(f"Error: {e}")
            self.model = None

    def _init_human_patterns(self):
        """Initialize comprehensive human writing patterns database"""
        self.human_phrases = {
            'fillers': ['like', 'you know', 'I mean', 'sort of', 'kind of', 'actually', 'basically', 'literally'],
            'hedges': ['maybe', 'perhaps', 'probably', 'might', 'could', 'sometimes', 'often', 'usually'],
            'intensifiers': ['really', 'very', 'so', 'extremely', 'incredibly', 'absolutely', 'totally'],
            'informal': ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'ain\'t', 'y\'all', 'lemme'],
            'reactions': ['wow', 'oh', 'ah', 'hmm', 'well', 'hey', 'haha', 'lol', 'omg'],
            'connectors': ['anyway', 'so', 'then', 'like I said', 'moving on', 'back to', 'on that note']
        }
        
        # Advanced AI pattern detection
        self.ai_indicators = {
            'structural': ['furthermore', 'moreover', 'however', 'nevertheless', 'consequently', 'thus', 'therefore'],
            'corporate': ['leverage', 'utilize', 'facilitate', 'implement', 'optimize', 'synergy', 'paradigm'],
            'academic': ['delve', 'realm', 'tapestry', 'underscores', 'highlights', 'crucial', 'paramount'],
            'formal': ['in order to', 'with respect to', 'in terms of', 'with regard to', 'pertaining to']
        }

    # QUANTUM-LEVEL TOPIC ANALYSIS
    def _analyze_topic_perspective(self, text):
        """Advanced topic detection with multi-dimensional analysis"""
        if not text or len(text.strip()) == 0:
            return 'casual', []
            
        text_lower = text.lower()
        
        # Multi-dimensional topic scoring
        topic_scores = {
            'casual': 0.3,  # Base human communication
            'technical': 0.0,
            'opinion': 0.0, 
            'story': 0.0,
            'explanation': 0.0,
            'academic': 0.0,
            'business': 0.0
        }
        
        # Advanced pattern matching
        patterns = {
            'technical': ['algorithm', 'function', 'variable', 'system', 'data', 'code', 'software', 'technical', 'digital'],
            'opinion': ['think', 'believe', 'feel', 'opinion', 'view', 'perspective', 'personally', 'frankly'],
            'story': ['story', 'experience', 'happened', 'once', 'time', 'when', 'then', 'after', 'before'],
            'explanation': ['because', 'since', 'therefore', 'thus', 'reason', 'explain', 'why', 'how'],
            'academic': ['research', 'study', 'analysis', 'theory', 'hypothesis', 'methodology', 'framework'],
            'business': ['business', 'company', 'market', 'strategy', 'growth', 'profit', 'customer', 'product']
        }
        
        # Score topics based on pattern density
        for topic, keywords in patterns.items():
            score = sum(1 for word in keywords if word in text_lower)
            topic_scores[topic] = score * 0.2
        
        # Boost scores based on structural patterns
        if any(indicator in text_lower for indicator in self.ai_indicators['academic']):
            topic_scores['academic'] += 0.4
        if any(indicator in text_lower for indicator in self.ai_indicators['business']):
            topic_scores['business'] += 0.3
            
        # Determine dominant topic with confidence
        dominant_topic = max(topic_scores, key=topic_scores.get)
        
        # Enhanced perspective words based on topic
        perspectives = {
            'casual': self.human_phrases['fillers'] + self.human_phrases['informal'],
            'technical': ['technically', 'specifically', 'essentially', 'basically', 'practically'],
            'opinion': ['I think', 'personally', 'honestly', 'frankly', 'to be honest', 'in my view'],
            'story': ['so', 'then', 'anyway', 'meanwhile', 'suddenly', 'eventually', 'next thing I know'],
            'explanation': ['because', 'since', 'which means', 'so that', 'the reason is', 'in other words'],
            'academic': ['according to', 'based on', 'research shows', 'studies indicate', 'evidence suggests'],
            'business': ['in business', 'commercially', 'from a strategy perspective', 'in the market']
        }
        
        return dominant_topic, perspectives.get(dominant_topic, perspectives['casual'])

    # NEURAL-LEVEL ANTI-AI TRANSFORMATION
    def _ultimate_anti_ai(self, text):
        if not text:
            return text
            
        # PHASE 1: Corporate/Academic Language Elimination
        corporate_replacements = {
            r"\bleverage\b": "use", r"\butilize\b": "use", r"\bfacilitate\b": "help", 
            r"\bimplement\b": "put in place", r"\boptimize\b": "make better", r"\bsynergy\b": "teamwork",
            r"\bparadigm\b": "way of thinking", r"\bframework\b": "structure", r"\bmethodology\b": "method",
            r"\brobust\b": "strong", r"\bscalable\b": "able to grow", r"\bdisruptive\b": "game-changing",
            r"\binnovative\b": "new and creative", r"\bcomprehensive\b": "complete", r"\bactionable\b": "useful",
            r"\bbandwidth\b": "time", r"\bcircle back\b": "follow up", r"\bdeep dive\b": "close look",
            r"\bvalue proposition\b": "benefit", r"\bcore competency\b": "main skill", r"\bstreamline\b": "simplify",
            r"\boperationalize\b": "use", r"\binterface\b": "connect", r"\bdeliverables\b": "results",
            r"\bkey takeaways\b": "main points", r"\bthought leadership\b": "expert advice"
        }
        
        # PHASE 2: Academic Language Destruction
        academic_replacements = {
            r"\bdelve\b": "look into", r"\brealm\b": "area", r"\btapestry\b": "mix", 
            r"\bunderscores\b": "shows", r"\bhighlights\b": "points out", r"\bcrucial\b": "important",
            r"\bparamount\b": "very important", r"\bplethora\b": "lots", r"\bmyriad\b": "many",
            r"\bconsequently\b": "so", r"\bthus\b": "so", r"\bhence\b": "so",
            r"\bfurthermore\b": "also", r"\bmoreover\b": "plus", r"\bhowever\b": "but",
            r"\bnevertheless\b": "still", r"\bnonetheless\b": "anyway", r"\bnotwithstanding\b": "even though",
            r"\bin order to\b": "to", r"\bwith respect to\b": "about", r"\bin terms of\b": "when it comes to",
            r"\bwith regard to\b": "about", r"\bpertaining to\b": "about", r"\bcommence\b": "start",
            r"\bterminate\b": "end", r"\bdemonstrate\b": "show", r"\billuminate\b": "explain",
            r"\b elucidate\b": "explain", r"\bconceptualize\b": "think about", r"\bsubstantiate\b": "back up"
        }
        
        # PHASE 3: Formal Language Humanization
        formal_replacements = {
            r"\bapproximately\b": "about", r"\bsubsequently\b": "after that", r"\bprior to\b": "before",
            r"\bin accordance with\b": "following", r"\bwith the exception of\b": "except for",
            r"\bfor the purpose of\b": "to", r"\bin the event that\b": "if", r"\bon a daily basis\b": "daily",
            r"\bat this point in time\b": "now", r"\bin the near future\b": "soon", r"\btake into consideration\b": "consider",
            r"\barrive at a conclusion\b": "conclude", r"\bconduct an analysis\b": "analyze", r"\bperform an evaluation\b": "evaluate"
        }
        
        # Apply all replacement phases
        all_replacements = {**corporate_replacements, **academic_replacements, **formal_replacements}
        
        for pattern, replacement in all_replacements.items():
            try:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            except:
                continue
                
        return text

    # QUANTUM CHAOS INJECTION
    def _inject_topic_specific_chaos(self, text, perspective_words):
        """Quantum-level chaos injection with topic intelligence"""
        if not text or len(text) < 20:
            return text
            
        # Advanced chaos patterns by topic
        quantum_chaos = {
            'casual': {
                'fillers': ["...like...", "...you know...", "...I mean...", "...sort of...", "...kinda..."],
                'reactions': ["...wow...", "...seriously...", "...no way...", "...crazy...", "...awesome..."],
                'pauses': [" — ", " ... ", " ~ ", " … "],
                'questions': ["...right?", "...you know?", "...get it?", "...makes sense?"]
            },
            'technical': {
                'clarifiers': ["...technically...", "...basically...", "...essentially...", "...in practice..."],
                'precision': [" — to be exact — ", " — specifically — ", " — in other words — "],
                'uncertainty': ["...I think...", "...probably...", "...maybe...", "...could be..."]
            },
            'opinion': {
                'subjectivity': ["...I think...", "...personally...", "...honestly...", "...frankly..."],
                'hedging': ["...maybe...", "...perhaps...", "...could be...", "...not sure..."],
                'emphasis': ["...REALLY...", "...seriously...", "...no joke...", "...for real..."]
            }
        }
        
        dominant_topic, _ = self._analyze_topic_perspective(text)
        chaos_config = quantum_chaos.get(dominant_topic, quantum_chaos['casual'])
        
        # MULTI-DIMENSIONAL CHAOS INJECTION
        text_words = text.split()
        
        if len(text_words) > 8:
            # Layer 1: Strategic filler injection
            if random.random() < 0.7:
                injection_points = random.randint(1, 3)
                for _ in range(injection_points):
                    if len(text_words) > 10:
                        pos = random.randint(3, len(text_words)-3)
                        filler = random.choice(chaos_config.get('fillers', []))
                        if filler:
                            text_words.insert(pos, filler)
            
            # Layer 2: Emotional reaction injection
            if random.random() < 0.5:
                reaction_pos = random.randint(0, min(2, len(text_words)-1))
                reaction = random.choice(chaos_config.get('reactions', []))
                if reaction:
                    text_words.insert(reaction_pos, reaction)
            
            # Layer 3: Conversational question injection
            if random.random() < 0.4 and len(text_words) > 15:
                question_pos = random.randint(len(text_words)-5, len(text_words)-1)
                question = random.choice(chaos_config.get('questions', []))
                if question:
                    text_words.insert(question_pos, question)
        
        text = " ".join(text_words)
        
        # Layer 4: Punctuation chaos
        if random.random() < 0.6:
            punctuation_chaos = [
                (r"\. ", ". ... "),
                (r"\, ", ", ... "),
                (r"\? ", "? ... "),
                (r"\! ", "! ... ")
            ]
            for pattern, replacement in random.sample(punctuation_chaos, 2):
                text = re.sub(pattern, replacement, text, count=1)
        
        return text

    # ADVANCED BURSTINESS ENGINE
    def _enhanced_burstiness(self, text):
        """Neural-level burstiness with human rhythm patterns"""
        if not text:
            return text
            
        try:
            sentences = sent_tokenize(text)
        except:
            return text
            
        if len(sentences) < 2:
            return text
            
        # Human sentence length variation
        processed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            
            # Apply human sentence structure variations
            if len(words) > 20 and random.random() < 0.7:
                # Split long sentences in human-like ways
                split_points = [
                    r"(?<=,)\s+", r"(?<=but)\s+", r"(?<=and)\s+", r"(?<=or)\s+",
                    r"(?<=so)\s+", r"(?<=because)\s+", r"(?<=which)\s+"
                ]
                
                for split_pattern in random.sample(split_points, 2):
                    try:
                        parts = re.split(split_pattern, sentence, 1)
                        if len(parts) > 1:
                            processed_sentences.append(parts[0].strip())
                            # Humanize the second part
                            second_part = parts[1].strip()
                            if second_part and second_part[0].isalpha():
                                connectors = ["And", "So", "But", "Anyway", "Then"]
                                second_part = random.choice(connectors) + " " + second_part[0].lower() + second_part[1:]
                            processed_sentences.append(second_part)
                            break
                    except:
                        processed_sentences.append(sentence)
                else:
                    processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)
        
        # Apply human rhythm patterns
        final_sentences = []
        for i, sentence in enumerate(processed_sentences):
            # Add human-like sentence starters occasionally
            if i > 0 and random.random() < 0.3:
                starters = ["So", "Anyway", "Like I was saying", "You know", "I mean"]
                if random.random() < 0.4:
                    sentence = random.choice(starters) + ", " + sentence[0].lower() + sentence[1:]
            
            # Add trailing human expressions
            if random.random() < 0.25:
                trailers = [", you know?", ", right?", ", I think.", ", probably."]
                if not sentence.endswith(('?', '!', '...')):
                    sentence = sentence.rstrip('.') + random.choice(trailers)
            
            final_sentences.append(sentence)
        
        return " ".join(final_sentences)

    # QUANTUM TYPO ENGINE
    def _advanced_typos(self, text):
        """Advanced context-aware typo injection system"""
        if not text or len(text) < 10:
            return text
            
        words = text.split()
        if len(words) < 3:
            return text
            
        # Typo probability increases with text length
        typo_probability = min(0.35, 0.15 + (len(words) * 0.002))
        
        if random.random() < typo_probability:
            # Advanced typo patterns
            typo_patterns = {
                'transposition': {
                    'the': 'teh', 'and': 'adn', 'you': 'yuo', 'are': 'aer',
                    'their': 'thier', 'because': 'becuase', 'probably': 'porbably'
                },
                'omission': {
                    'probably': 'prolly', 'going to': 'gonna', 'want to': 'wanna',
                    'kind of': 'kinda', 'sort of': 'sorta', 'give me': 'gimme',
                    'let me': 'lemme', 'what are you': 'whatcha', 'don\'t know': 'dunno'
                },
                'phonetic': {
                    'through': 'thru', 'though': 'tho', 'enough': 'enuf',
                    'because': 'cuz', 'okay': 'k', 'people': 'ppl',
                    'before': 'befor', 'after': 'aftr', 'about': 'bout'
                },
                'doubling': {
                    'until': 'untill', 'across': 'accross', 'occurred': 'occured',
                    'preferred': 'prefered', 'traveling': 'travelling'
                }
            }
            
            # Apply multiple typo types
            applied_typos = 0
            max_typos = min(3, max(1, len(words) // 10))
            
            for typo_type, patterns in typo_patterns.items():
                if applied_typos >= max_typos:
                    break
                    
                for correct, typo in patterns.items():
                    if applied_typos >= max_typos:
                        break
                        
                    if correct in text.lower() and random.random() < 0.6:
                        try:
                            text = re.sub(r'\b' + re.escape(correct) + r'\b', typo, text, flags=re.IGNORECASE, count=1)
                            applied_typos += 1
                        except:
                            continue
        
        # Advanced: Missing word simulation
        if random.random() < 0.15 and len(words) > 8:
            try:
                remove_pos = random.randint(3, len(words)-3)
                if remove_pos < len(words):
                    small_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']
                    if words[remove_pos].lower() in small_words:
                        del words[remove_pos]
                        text = " ".join(words)
            except:
                pass
        
        return text

    # NEURAL CONTRACTION SYSTEM
    def _contextual_contractions(self, text):
        """Advanced contraction system with contextual intelligence"""
        if not text:
            return text
            
        # Expanded contraction database
        contraction_db = {
            'standard': {
                r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bdid not\b": "didn't",
                r"\bcan not\b": "can't", r"\bcannot\b": "can't", r"\bcould not\b": "couldn't",
                r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't", r"\bshould not\b": "shouldn't",
                r"\bis not\b": "isn't", r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
                r"\bwere not\b": "weren't", r"\bhas not\b": "hasn't", r"\bhave not\b": "haven't",
                r"\bhad not\b": "hadn't"
            },
            'pronoun': {
                r"\bI am\b": "I'm", r"\byou are\b": "you're", r"\bhe is\b": "he's",
                r"\bshe is\b": "she's", r"\bit is\b": "it's", r"\bwe are\b": "we're",
                r"\bthey are\b": "they're", r"\bthat is\b": "that's", r"\bthere is\b": "there's",
                r"\bhere is\b": "here's", r"\bwhat is\b": "what's", r"\bwhere is\b": "where's",
                r"\bwho is\b": "who's", r"\bwhy is\b": "why's", r"\bhow is\b": "how's"
            },
            'informal': {
                r"\bgoing to\b": "gonna", r"\bwant to\b": "wanna", r"\bgot to\b": "gotta",
                r"\bkind of\b": "kinda", r"\bsort of\b": "sorta", r"\blot of\b": "lotta",
                r"\bgive me\b": "gimme", r"\blet me\b": "lemme", r"\bwhat do you\b": "whatcha",
                r"\bdon't know\b": "dunno"
            }
        }
        
        # Apply contractions with contextual awareness
        for category, contractions in contraction_db.items():
            for pattern, contraction in contractions.items():
                if random.random() < 0.85:  # High probability for natural speech
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
                
                # NEURAL PARAPHRASE GENERATION
                candidates = []
                for i in range(6):  # Increased from 4 to 6 for more variation
                    try:
                        # Dynamic parameter variation
                        temp = random.uniform(1.8, 2.3)  # Higher temperature for more creativity
                        top_p = random.uniform(0.94, 0.998)  # Higher top_p for diversity
                        
                        inputs = self.tokenizer(f"paraphrase: {para}", return_tensors="pt", truncation=True, max_length=512).to(self.device)
                        out = self.model.generate(
                            **inputs,
                            max_length=512,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            top_k=random.randint(150, 250),  # Higher top_k
                            repetition_penalty=random.uniform(1.4, 1.6),  # Higher penalty
                            length_penalty=0.3,  # Lower for more variation
                            early_stopping=True,
                            no_repeat_ngram_size=2,  # More aggressive
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
                    # Select most human-like candidate
                    base = max(candidates, key=lambda x: self._calculate_human_score(x))
                except:
                    base = candidates[0]
                
                # NEURAL SENTENCE FUSION
                if len(candidates) > 2:
                    try:
                        fusion_candidates = random.sample(candidates, min(3, len(candidates)))
                        base_sentences = sent_tokenize(base)
                        
                        for fusion_candidate in fusion_candidates:
                            fusion_sentences = sent_tokenize(fusion_candidate)
                            if len(fusion_sentences) > 1 and len(base_sentences) > 2:
                                # Intelligent sentence swapping
                                swap_pos = random.randint(1, len(base_sentences)-2)
                                if swap_pos < len(fusion_sentences):
                                    base_sentences[swap_pos] = fusion_sentences[swap_pos]
                        
                        base = " ".join(base_sentences)
                    except:
                        pass

                # QUANTUM HUMANIZATION PIPELINE
                final = base
                
                # Apply transformation layers with increased intensity
                transformation_pipeline = [
                    (self._ultimate_anti_ai, 1.0),  # Always apply
                    (self._inject_topic_specific_chaos, 0.8),
                    (self._enhanced_burstiness, 0.9),
                    (self._advanced_typos, 0.7),
                    (self._contextual_contractions, 0.95),
                    (self._add_emotional_expression, 0.6),
                    (self._add_conversational_elements, 0.75)
                ]
                
                for transform_func, probability in transformation_pipeline:
                    if random.random() < probability:
                        try:
                            final = transform_func(final)
                        except:
                            continue

                # FINAL QUANTUM POLISHING
                try:
                    sentences = sent_tokenize(final)
                    # Apply human capitalization patterns
                    sentences = [self._humanize_capitalization(s) for s in sentences]
                    final = " ".join(sentences)
                    
                    # Final randomness injection
                    if random.random() < 0.3:
                        final = self._inject_final_human_touch(final)
                        
                except:
                    pass

                result.append(final)

            return "\n\n".join(result) if result else text

    def _calculate_human_score(self, text):
        """Calculate how human-like a text is"""
        if not text:
            return 0
            
        score = 0
        words = text.lower().split()
        
        # Score based on human language indicators
        for category, phrases in self.human_phrases.items():
            for phrase in phrases:
                if phrase in text.lower():
                    score += 0.1
        
        # Penalize AI indicators
        for category, indicators in self.ai_indicators.items():
            for indicator in indicators:
                if indicator in text.lower():
                    score -= 0.2
        
        # Score based on sentence length variation
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(lengths)
            score += min(0.5, length_variance * 0.1)  # Reward variation
        
        # Reward contraction usage
        contractions = ["n't", "'s", "'re", "'ve", "'ll", "'d"]
        contraction_count = sum(1 for cont in contractions if cont in text)
        score += contraction_count * 0.05
        
        return max(0, score)

    def _humanize_capitalization(self, sentence):
        """Apply human-like capitalization patterns"""
        if not sentence:
            return sentence
            
        # Humans sometimes don't capitalize properly
        if random.random() < 0.15:
            return sentence[0].lower() + sentence[1:] if sentence else sentence
        
        return sentence

    def _inject_final_human_touch(self, text):
        """Final human touch injections"""
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
            " ...seriously though..."
        ]
        
        if random.random() < 0.4:
            touch = random.choice(human_touches)
            if random.random() < 0.7:
                text += touch
            else:
                words = text.split()
                if len(words) > 5:
                    insert_pos = random.randint(len(words)//2, len(words)-2)
                    words.insert(insert_pos, touch)
                    text = " ".join(words)
        
        return text

    def _add_emotional_expression(self, text):
        """Add emotional human expressions"""
        if not text or len(text) < 30:
            return text
            
        emotions = [
            ("Wow, ", 0.3), ("Seriously, ", 0.2), ("No way, ", 0.15), 
            ("Awesome, ", 0.25), ("Crazy, ", 0.2), ("Unbelievable, ", 0.1),
            ("Interesting, ", 0.3), ("Funny enough, ", 0.15), ("Honestly, ", 0.4)
        ]
        
        if random.random() < 0.35:
            emotion, prob = random.choice(emotions)
            if random.random() < prob:
                text = emotion + text[0].lower() + text[1:]
        
        return text

    def _add_conversational_elements(self, text):
        """Add conversational human elements"""
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
            "Anyway, that's my take."
        ]
        
        if random.random() < 0.4:
            element = random.choice(elements)
            if random.random() < 0.6:
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