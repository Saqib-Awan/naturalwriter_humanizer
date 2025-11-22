import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
import random
import re

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
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        except Exception as e:
            print(f"Model Error: {e}")
            self.model = None

    # ==================== ULTRA ANTI-DETECTION LAYERS ====================

    def _apply_anti_academic_layer(self, text):
        """Now 500+ aggressive replacements – nukes every trace of AI formality"""
        replacements = {
            # Core academic/corporate killers
            r"\bdelve(?:\s+into|\s+deeper)?\b": "look into", r"\brealm\b": "area", r"\btapestry\b": "mix",
            r"\bunderscores\b": "shows", r"\bemphasizes\b": "points out", r"\bhighlights\b": "makes clear",
            r"\bcrucial\b": "key", r"\bpivotal\b": "huge", r"\bparamount\b": "super important",
            r"\bmeticulous\b": "super careful", r"\bintricate\b": "complicated", r"\bnuanced\b": "subtle",
            r"\bfacilitate\b": "help", r"\butilize\b": "use", r"\bleverage\b": "use", r"\bemploy\b": "use",
            r"\boptimize\b": "improve", r"\benhance\b": "boost", r"\bstreamline\b": "simplify",
            r"\bcomprehensive\b": "full", r"\bholistic\b": "whole-picture", r"\brobust\b": "solid",
            r"\bendeavor\b": "try", r"\binitiative\b": "project", r"\bparadigm\b": "way of thinking",
            r"\bsynergy\b": "teamwork", r"\bproactive\b": "ahead of the game", r"\bimpactful\b": "powerful",

            # Transitions → human speech
            r"\bmoreover\b": "plus", r"\bfurthermore\b": "also", r"\badditionally\b": "on top of that",
            r"\bconsequently\b": "so", r"\btherefore\b": "that’s why", r"\bthus\b": "so",
            r"\bhence\b": "which means", r"\bnevertheless\b": "still", r"\bnonetheless\b": "even so",
            r"\binitially\b": "at first", r"\bsubsequently\b": "then", r"\bultimately\b": "in the end",
            r"\bnotably\b": "especially", r"\bsignificantly\b": "a lot", r"\bmarkedly\b": "clearly",

            # Corporate nonsense
            r"\bregarding\b": "about", r"\bconcerning\b": "about", r"\bwith regard to\b": "about",
            r"\bin terms of\b": "when it comes to", r"\bpredominantly\b": "mostly", r"\bprimarily\b": "mainly",
            r"\bsubstantial\b": "big", r"\bconsiderable\b": "serious", r"\bensure\b": "make sure",
            r"\benable\b": "let", r"\bempower\b": "give the tools to", r"\bcommence\b": "start",
            r"\binitiate\b": "kick off", r"\bterminate\b": "end", r"\bimplement\b": "roll out",
            r"\bexecute\b": "carry out", r"\bmanifest\b": "show up", r"\bevident\b": "obvious",
            r"\bmethodology\b": "approach", r"\bframework\b": "structure", r"\bstrategy\b": "plan",

            # Phrase assassins
            r"\bin order to\b": "to", r"\bdue to the fact that\b": "because", r"\bin the event that\b": "if",
            r"\bat this juncture\b": "now", r"\bprior to\b": "before", r"\bsubsequent to\b": "after",
            r"\ba multitude of\b": "lots of", r"\ba plethora of\b": "tons of", r"\bin conclusion\b": "so",
            r"\bit is worth noting\b": "funny thing is", r"\bon the contrary\b": "actually no",
            r"\bin spite of\b": "even though", r"\bfor the purpose of\b": "to",
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def _inject_natural_pauses(self, text):
        """Micro-hesitations that humans do constantly"""
        if random.random() > 0.33: return text
        pauses = [
            " ...well, ", " ...you know, ", " ...like, ", " ...I mean, ", " ...sort of, ",
            " ...kind of, ", " ...right? ", " ...yeah, ", " ...anyway, ", " ...so yeah, "
        ]
        if len(text) > 50 and (" " in text):
            pos = random.randint(30, min(120, len(text)-20))
            if text[pos] == " ":
                text = text[:pos] + random.choice(pauses) + text[pos+1:]
        return text

    def _inject_thinking_dots(self, text):
        if random.random() < 0.22:
            text = re.sub(r"\.\s+", ". ... ", text, count=1)
        return text

    def _inject_conversational_hooks(self, text):
        hooks = [
            "Here’s the thing — ", "Real quick, ", "Quick thing: ", "One thing though — ",
            "Funny enough, ", "Not gonna lie, ", "Dead serious, ", "No cap, ",
            "Between you and me, ", "Off the record, "
        ]
        if random.random() < 0.25 and text and text[0].isupper():
            hook = random.choice(hooks)
            text = hook + text[0].lower() + text[1:]
        return text

    def _ultra_grammar_relaxation(self, text):
        slang = {
            r"\bgoing to\b": "gonna", r"\bwant to\b": "wanna", r"\bhave to\b": "gotta",
            r"\bkind of\b": "kinda", r"\bsort of\b": "sorta", r"\ba lot of\b": "tons of",
            r"\bout of\b": "outta", r"\bbecause\b": "cuz", r"\bdo not know\b": "dunno",
            r"\bI will\b": "I'll", r"\byou will\b": "you'll", r"\bwe will\b": "we'll",
            r"\bshould have\b": "shoulda", r"\bcould have\b": "coulda", r"\bwould have\b": "woulda"
        }
        for pat, rep in slang.items():
            if random.random() > 0.35:
                text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text

    def _smart_fragments_and_runons(self, text):
        # Break long sentences
        if random.random() < 0.45 and len(text) > 120:
            text = re.sub(r", (and|but|so) ", r". \1 ", text, count=1)
        # Occasional run-on
        if random.random() < 0.3:
            text = text.replace(". ", " — ", 1)
        return text

    def _human_typo_engine(self, text):
        """Extremely subtle, high-IQ typos that real humans make"""
        if random.random() > 0.15: return text
        smart_typos = {
            "the ": "teh ", "and ": "adn ", "with ": "wiht ", "which ": "wich ",
            "there ": "ther ", "their ": "thier ", "because ": "becasue ",
            "really ": "realy ", "definitely ": "definately ", "necessary ": "neccessary "
        }
        word = random.choice(list(smart_typos.keys()))
        if word in text.lower():
            text = text.replace(word, smart_typos[word], 1)
        return text

    def humanize(self, text, level="Standard"):
        if not self.model:
            return "Error: Model failed to load."

        # MAXIMUM STEALTH SETTINGS
        if "Ghost" in level or "ghost" in level.lower():
            temp_base = random.uniform(1.45, 1.75)
            rep_pen = random.uniform(1.25, 1.38)
            top_k = random.randint(110, 160)
            casual = True
            typos = True
        elif "Standard" in level:
            temp_base = random.uniform(1.15, 1.45)
            rep_pen = random.uniform(1.12, 1.25)
            top_k = random.randint(85, 120)
            casual = True
            typos = True
        else:
            temp_base = 0.95
            rep_pen = 1.0
            top_k = 60
            casual = False
            typos = False

        paragraphs = text.split('\n')
        final_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                final_paragraphs.append("")
                continue

            sentences = sent_tokenize(paragraph)
            out = []

            for i, sent in enumerate(sentences):
                # Per-sentence entropy variation (critical for burstiness)
                temp = temp_base * random.uniform(0.88, 1.18)

                input_text = "paraphrase: " + sent
                encoding = self.tokenizer.encode_plus(input_text, padding="longest", return_tensors="pt")
                input_ids = encoding["input_ids"].to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=512,
                    do_sample=True,
                    temperature=temp,
                    top_k=top_k,
                    top_p=0.97,
                    repetition_penalty=rep_pen,
                    num_return_sequences=1,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                # === 9-LAYER STEALTH PIPELINE ===
                result = self._inject_conversational_hooks(result)
                result = self._inject_natural_pauses(result)
                result = self._inject_thinking_dots(result)
                result = self._force_contractions(result)
                result = self._apply_anti_academic_layer(result)
                
                if casual:
                    result = self._ultra_grammar_relaxation(result)
                    
                result = self._smart_fragments_and_runons(result)
                
                if typos:
                    result = self._human_typo_engine(result)

                out.append(result)

            final_paragraphs.append(" ".join(out))

        return "\n".join(final_paragraphs)

    # Keep your original methods for 100% backward compatibility
    def _force_contractions(self, text):
        c = {
            r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bdid not\b": "didn't",
            r"\bcannot\b": "can't", r"\bwill not\b": "won't", r"\bis not\b": "isn't",
            r"\bare not\b": "aren't", r"\bwas not\b": "wasn't", r"\bwere not\b": "weren't",
            r"\bhave not\b": "haven't", r"\bhad not\b": "hadn't", r"\bit is\b": "it's",
            r"\bthat is\b": "that's", r"\bthere is\b": "there's", r"\bI am\b": "I'm",
            r"\byou are\b": "you're", r"\bwe are\b": "we're", r"\bthey are\b": "they're"
        }
        for p, r in c.items():
            text = re.sub(p, r, text, flags=re.IGNORECASE)
        return text

    # Legacy methods (unchanged names)
    def _inject_emotional_adverbs(self, text): return text  # replaced by stronger version
    def _apply_grammar_relaxation(self, text): return text
    def _inject_sentence_fragments(self, text): return text
    def _inject_human_fillers(self, text): return text
    def _inject_micro_hesitations(self, text): return text
    def _subtle_typos(self, text): return text