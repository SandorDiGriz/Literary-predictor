from pymorphy2 import MorphAnalyzer
from functools import lru_cache
from tqdm import tqdm
from lexical_diversity import lex_div as ld

import nltk
import numpy as np
import re
import math
import random
import csv


class Extractor:
    def get_text(self, path):
        with open(path, "r", encoding="utf-8") as file:
            text = file.readlines()

        return text

    def tokenize(self, text, regex="[А-Яа-яёA-z]+"):
        regex = re.compile(regex)
        tokens = regex.findall(text.lower())

        return tokens

    def remove_stopwords(
        self, lemmas, stopwords=nltk.corpus.stopwords.words("russian")
    ):
        return [w for w in lemmas if not w in stopwords and len(w) > 3]

    @lru_cache(maxsize=128)
    def lemmatize_word(self, token):
        pymorphy = MorphAnalyzer()
        return pymorphy.parse(token)[0].normal_form

    def lemmatize_text(self, text):
        return [self.lemmatize_word(w) for w in tqdm(text)]

    def clean_text(self, text):
        tokens = self.tokenize("".join(text))
        lemmas = self.lemmatize_text(tokens)

        return self.remove_stopwords(lemmas)

    def avg_sent_len(self, text, cut_edge=False):
        sentences = nltk.sent_tokenize("".join(text))
        sent_len = [len(self.tokenize(sent)) for sent in sentences]
        if len(sent_len) > 20 and cut_edge:
            sent_len = sent_len[20:]

        return np.mean(sent_len)

    def avg_word_len(self, text, tokenize=True, cut_edge=False):
        if tokenize:
            tokens = self.tokenize("".join(text))
        else:
            tokens = text
        word_len = [len(token) for token in tokens]
        if len(word_len) > 500 and cut_edge:
            word_len = word_len[500:]

        return np.mean(word_len)

    def avg_words_per_prg(self, text, tokenize=True, cut_edge=False):
        if tokenize:
            tokens = self.tokenize("".join(text))
        else:
            tokens = text
        paragraphs = list(filter(lambda x: x != "", "".join(text).split("\n\n")))
        if len(paragraphs) > 100 and cut_edge:
            paragraphs = paragraphs[100:]

        return len(tokens) / len(paragraphs)

    def char_freq(self, text, char):
        tokens = nltk.word_tokenize("".join(text))
        char_distfreq = nltk.probability.FreqDist(tokens)

        return (char_distfreq[str(char)] * 1000) / char_distfreq.N()

    def lexical_complexity(
        self, text, path="src/feature-extractor/vocab by level.csv", tokenize=True
    ):
        if tokenize:
            tokens = self.tokenize("".join(text))
        else:
            tokens = text
        lex_level = []
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            dict = file.readlines()
            sample_tokens = self.lemmatize_text(random.choices(tokens, k=1000))

            for lemma in sample_tokens:
                for line in dict:
                    values = line.split(",")
                    if lemma == values[0]:
                        lex_level.append(values[1].split("\n")[0])

            voc_percentage = {
                "A1": round(lex_level.count("1E") / 1000, 2),
                "A2": round(lex_level.count("2I") / 1000, 2),
                "B1": round(lex_level.count("3A") / 1000, 2),
                "B2": round(lex_level.count("3AU") / 1000, 2),
                "C1": round(lex_level.count("4S") / 1000, 2),
                "C2": round(lex_level.count("4SU") / 1000, 2),
                "unknown": round((1000 - len(lex_level)) / 1000, 2),
            }

            return voc_percentage

    def ttr(self, text, mode="standard", tokenize=True, lemmatize=False):
        if tokenize:
            tokens = self.tokenize("".join(text))
        else:
            tokens = text
        if lemmatize:
            tokens = self.lemmatize_text(tokens)
        if mode == "standard":
            return len(set(tokens)) / len(tokens)

        elif mode == "root":
            return len(set(tokens)) / math.sqrt(len(tokens))

        elif mode == "corrected":
            return len(set(tokens)) / math.sqrt(2 * len(tokens))

        elif mode == "log":
            return math.log10(len(set(tokens))) / math.log10(len(tokens))

        elif mode == "hdd":
            return ld.hdd(tokens)

        elif mode == "mtld":
            return ld.mtld(tokens)

        else:
            raise ValueError(
                f"Current mode is '{mode}' but should be in ('standard', 'root', 'corrected')"
            )

    def readability_score_oborneva(self, text, method="fre"):
        syllables = len(
            re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", " ".join(text))
        )
        words = len(self.tokenize("".join(text)))
        sentences = len(nltk.sent_tokenize("".join(text)))
        if method == "fre":
            return 206.835 - 1.3 * (words / sentences) - 60.1 * (syllables / words)
        elif method == "fkd":
            return (0.5 * words / sentences) + (8.4 * syllables / words) - 15.59
        else:
            raise ValueError(
                f"Current method is '{method}' but should be in ('fre', 'fkd')"
            )

    def readability_score_soloviev(self, text, method="fre"):
        syllables = len(
            re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", " ".join(text))
        )
        words = len(self.tokenize("".join(text)))
        sentences = len(nltk.sent_tokenize("".join(text)))
        if method == "fre":
            return 208.7 - 2.6 * (words / sentences) - 39.2 * (syllables / words)
        elif method == "fkd":
            return (0.36 * words / sentences) + (5.76 * syllables / words) - 11.97
        else:
            raise ValueError(
                f"Current method is '{method}' but should be in ('fre', 'fkd')"
            )

    def extract_all(self, text, cut_edge=False):
        lex_complx = self.lexical_complexity(text)
        methods = {
            "avg_word_len": self.avg_word_len(text, cut_edge, tokenize=True),
            "avg_sent_len": self.avg_sent_len(text, cut_edge),
            "avg_words_per_par": self.avg_words_per_prg(text, cut_edge, tokenize=True),
            "comma_freq": self.char_freq(text, ","),
            "colon_freq": self.char_freq(text, ":"),
            "dash_freq": self.char_freq(text, "—"),
            "A1 voc": lex_complx.get("A1"),
            "A2 voc": lex_complx.get("A2"),
            "B1 voc": lex_complx.get("B1"),
            "B2 voc": lex_complx.get("B2"),
            "C1 voc": lex_complx.get("C1"),
            "C2 voc": lex_complx.get("C2"),
            "unknown voc": lex_complx.get("unknown"),
            "TTR": self.ttr(text),
            "TTR-root": self.ttr(text, "root"),
            "TTR-log": self.ttr(text, "log"),
            "TTR-corrected": self.ttr(text, "corrected"),
            "TTR-hdd": self.ttr(text, "hdd"),
            "TTR-mtld": self.ttr(text, "mtld"),
            "FKD-oborneva": self.readability_score_oborneva(text, "fkd"),
            "FRE-oborneva": self.readability_score_oborneva(text, "fre"),
            "FKD-soloviev": self.readability_score_soloviev(text, "fkd"),
            "FRE-soloviev": self.readability_score_soloviev(text, "fre"),
        }

        return methods


et = Extractor()
et_t = et.get_text(
    "src/corpus txt/Человек для особых поручений - Антон В. Демченко.txt"
)
et_all = et.extract_all(et_t)
for i in et_all:
    print(i, et_all[i])

# cut edge
