import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import random
import re
import streamlit as st

# --- NLTK SETUP ---
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_resources()

class Humanizer:
    def __init__(self):
        self.model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        self.device = "cpu"
        
        print(f"Loading TITAN T5 Model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model failed to load: {e}")
            self.model = None
            self.tokenizer = None

    def safe_choice(self, items, fallback=""):
        """Never crash on empty list"""
        if not items:
            return fallback
        try:
            return random.choice(items)
        except (ValueError, IndexError):
            return fallback

    def _apply_anti_academic_layer(self, text):
        if not text.strip():
            return text
            
        synonyms = {
            r"\bdelve\s+(deeper\s+)?into\b": "dig into", r"\brealm\b": "area", r"\btapestry\b": "mix",
            r"\bunderscores\b": "shows", r"\bhighlights\b": "points out", r"\bcrucial\b": "key",
            r"\bparamount\b": "huge", r"\bmeticulous\b": "careful", r"\bintricate\b": "tricky",
            r"\bfacilitate\b": "help", r"\butilize\b": "use", r"\bleverage\b": "use",
            r"\boptimize\b": "improve", r"\bcomprehensive\b": "full", r"\bholistic\b": "whole",
            r"\bendeavor\b": "try", r"\bparadigm\b": "model", r"\bsynergy\b": "teamwork",
            r"\bproactive\b": "active", r"\bimpactful\b": "strong", r"\btransformative\b": "game-changing",
            r"\bmoreover\b": "plus", r"\bfurthermore\b": "also", r"\badditionally\b": "also",
            r"\bconsequently\b": "so", r"\btherefore\b": "so", r"\bthus\b": "so",
            r"\bhence\b": "that's why", r"\bnevertheless\b": "still", r"\bnonetheless\b": "even so",
            r"\binitially\b": "first", r"\bsubsequently\b": "then", r"\bultimately\b": "in the end",
            r"\bnotably\b": "especially", r"\bsignificantly\b": "a lot", r"\bconsiderably\b": "quite",
            r"\bregarding\b": "about", r"\bconcerning\b": "about", r"\bwith respect to\b": "about",
            r"\bin terms of\b": "when it comes to", r"\bpredominantly\b": "mostly", r"\bprimarily\b": "mainly",
            r"\bin order to\b": "to", r"\bdue to the fact that\b": "because", r"\bin the event that\b": "if",
            r"\ba number of\b": "some", r"\ba variety of\b": "different", r"\ba plethora of\b": "tons of",
        }

        for pattern, replacement in synonyms.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _inject_micro_hesitations(self, text):
        if random.random() > 0.35 or len(text) < 30 or ", " not in text:
            return text
        fillers = [" you know,", " I mean,", " like,", " kinda,", " sort of,", " well,", " anyway,", " right?", " yeah,"]
        pos = text.find(", ")
        if pos <= 20 or pos == -1:
            return text
        chosen = self.safe_choice(fillers)
        if chosen:
            text = text[:pos + 2] + chosen + text[pos + 2:]
        return text

    def _inject_emotional_adverbs(self, text):
        if random.random() > 0.35 or len(text) < 20 or ", " not in text:
            return text
        adverbs = [" honestly,", " frankly,", " seriously,", " thankfully,", " luckily,", " sadly,", " weirdly,", " surprisingly,", " obviously,", " totally,"]
        chosen = self.safe_choice(adverbs)
        if chosen:
            text = text.replace(", ", chosen + " ", 1)
        return text

    def _apply_grammar_relaxation(self, text):
        if len(text) < 15:
            return text
        replacements = {
            r"\bgoing to\b": "gonna", r"\bwant to\b": "wanna", r"\bhave to\b": "gotta",
            r"\bkind of\b": "kinda", r"\bsort of\b": "sorta", r"\ba lot of\b": "tons of",
            r"\bdo not know\b": "dunno", r"\bI will\b": "I'll", r"\byou will\b": "you'll",
        }
        for pattern, repl in replacements.items():
            if random.random() > 0.4:
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def _inject_sentence_fragments(self, text):
        text = re.sub(r"\.\s+However,", ". But,", text, flags=re.IGNORECASE)
        text = re.sub(r"\.\s+Therefore,", ". So,", text, flags=re.IGNORECASE)
        if random.random() < 0.4 and ", and " in text.lower():
            text = text.replace(", and ", ". And ", 1)
        if random.random() < 0.15:
            text = text.replace(" because ", " cuz ")
        return text

    def _inject_human_fillers(self, text):
        if len(text) < 20:
            return text
        fillers = ["Look, ", "Listen, ", "Honestly, ", "Real talk, ", "Basically, ", "You see, ", "Thing is, ", "Anyway, "]
        if random.random() < 0.3:
            filler = self.safe_choice(fillers)
            if filler and text.strip():
                text = filler + text[0].lower() + text[1:]
        return text

    def _force_contractions(self, text):
        contractions = {
            r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bis not\b": "isn't",
            r"\bare not\b": "aren't", r"\bcannot\b": "can't", r"\bwill not\b": "won't",
            r"\bit is\b": "it's", r"\bthat is\b": "that's", r"\bI am\b": "I'm",
            r"\byou are\b": "you're", r"\bwe are\b": "we're", r"\bthey are\b": "they're"
        }
        for pattern, repl in contractions.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def _subtle_typos(self, text):
        if random.random() > 0.15 or len(text) < 30:
            return text
        typos = { "the ": "teh ", "and ": "adn ", "with ": "wiht ", "because ": "becuase " }
        word = self.safe_choice(list(typos.keys()))
        if word and word in text.lower():
            text = re.sub(word, typos[word], text, flags=re.IGNORECASE, count=1)
        return text

    def humanize(self, text, level="Standard"):
        if not self.model or not self.tokenizer:
            return "Error: Paraphraser model not available."

        if not text.strip():
            return text

        # Settings based on level
        if "Ghost" in level:
            temperature = random.uniform(1.2, 1.5)
            top_k = random.randint(80, 120)
            do_casual = True
            allow_typos = True
        else:
            temperature = random.uniform(0.9, 1.2)
            top_k = random.randint(60, 90)
            do_casual = True
            allow_typos = False

        paragraphs = [p for p in text.split('\n') if p.strip() or p == ""]
        final_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                final_paragraphs.append("")
                continue

            try:
                sentences = sent_tokenize(paragraph)
                if not sentences:
                    final_paragraphs.append(paragraph)
                    continue

                humanized_sentences = []

                for sentence in sentences:
                    if len(sentence) < 10:
                        humanized_sentences.append(sentence)
                        continue

                    input_text = "paraphrase: " + sentence.strip()
                    try:
                        encoding = self.tokenizer.encode_plus(
                            input_text,
                            padding="longest",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt"
                        )
                        input_ids = encoding["input_ids"].to(self.device)

                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                max_length=256,
                                do_sample=True,
                                top_k=top_k,
                                top_p=0.95,
                                temperature=max(0.7, temperature * random.uniform(0.9, 1.1)),
                                repetition_penalty=1.1,
                                num_return_sequences=1,
                                early_stopping=True
                            )

                        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                        # Critical: Fallback if model failed
                        if not result or len(result) < 5 or "paraphrase:" in result.lower():
                            result = sentence

                    except Exception as e:
                        print(f"Generation failed: {e}")
                        result = sentence  # fallback

                    # Post-processing pipeline
                    result = self._inject_human_fillers(result)
                    result = self._inject_emotional_adverbs(result)
                    result = self._inject_micro_hesitations(result)
                    result = self._force_contractions(result)
                    if do_casual:
                        result = self._apply_grammar_relaxation(result)
                    result = self._inject_sentence_fragments(result)
                    result = self._apply_anti_academic_layer(result)
                    if allow_typos and "Ghost" in level:
                        result = self._subtle_typos(result)

                    humanized_sentences.append(result)

                final_paragraphs.append(" ".join(humanized_sentences))

            except Exception as e:
                print(f"Paragraph processing failed: {e}")
                final_paragraphs.append(paragraph)  # safe fallback

        return "\n".join(final_paragraphs)


# ============== STREAMLIT APP (Optional) ==============
# Uncomment below if you're deploying to Streamlit

"""
st.set_page_config(page_title="AI Text Humanizer", layout="centered")

st.title("AI Text Humanizer")
st.write("Turn robotic AI text into natural, human-sounding writing.")

# Initialize humanizer with caching
@st.cache_resource
def load_humanizer():
    return Humanizer()

humanizer = load_humanizer()

input_text = st.text_area("Paste your text here:", height=200)
level = st.selectbox("Style Level:", ["Standard", "Ghost Mode (Very Human)"])

if st.button("Humanize Text"):
    if input_text.strip():
        with st.spinner("Making it sound human..."):
            result = humanizer.humanize(input_text, level=level)
        st.success("Done!")
        st.write("### Humanized Version:")
        st.write(result)
    else:
        st.warning("Please enter some text!")
"""

# Usage:
# humanizer = Humanizer()
# print(humanizer.humanize("Your AI text here...", level="Ghost"))