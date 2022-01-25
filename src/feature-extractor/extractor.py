from pymorphy2 import MorphAnalyzer
from functools import lru_cache
from tqdm import tqdm

import nltk
import numpy as np
import re
import math
import random


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

    def avg_word_len(self, text, cut_edge=False):
        tokens = self.tokenize("".join(text))  # set?
        word_len = [len(token) for token in tokens]
        if len(word_len) > 500 and cut_edge:
            word_len = word_len[500:]

        return np.mean(word_len)

    def avg_words_per_prg(self, text, cut_edge=False):
        tokens = self.tokenize("".join(text))
        paragraphs = list(filter(lambda x: x != "", "".join(text).split("\n\n")))
        if len(paragraphs) > 100 and cut_edge:
            paragraphs = paragraphs[100:]

        return len(tokens) / len(paragraphs)

    def char_freq(self, text, char):
        tokens = nltk.word_tokenize("".join(text))
        char_distfreq = nltk.probability.FreqDist(tokens)

        return (char_distfreq[str(char)] * 1000) / char_distfreq.N()

    def lexical_complexity(
        self, text, stopwords=nltk.corpus.stopwords.words("russian")
    ):
        tokens = self.tokenize("".join(text))
        lemmas = []
        counter = 0
        pbar = tqdm(total=1000)

        with open(
            "/Users/apotekhin/repositories/literary-predictor/src/feature-extractor/simple lex.txt",
            "r",
            encoding="utf-8",
        ) as file:
            simple_lex = file.readlines()[0].split()
            while len(lemmas) < 1000:
                sample_token = self.lemmatize_word(random.choice(tokens))
                lemma = sample_token if not sample_token in stopwords else []

                if lemma:
                    pbar.update(1)
                    lemmas.append("".join(lemma))
                    if lemma in simple_lex:
                        counter += 1
            pbar.close()

            return counter / 1000

    def ttr(self, text, mode="standard", lemmatize=False):
        tokens = self.tokenize("".join(text))
        # not lemmas ((
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

        else:
            raise ValueError(
                f"Current mode is '{mode}' but should be in ('standard', 'root', 'corrected')"
            )

    def readability_score_oborneva(self, text, func="fre"):
        syllables = len(
            re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", " ".join(text))
        )
        words = len(self.tokenize("".join(text)))
        sentences = len(nltk.sent_tokenize("".join(text)))
        if func == "fre":
            return 206.835 - 1.3 * (words / sentences) - 60.1 * (syllables / words)
        elif func == "fkd":
            return (0.5 * words / sentences) + (8.4 * syllables / words) - 15.59
        else:
            raise ValueError(
                f"Current function is '{func}' but should be in ('fre', 'fkd')"
            )

    def readability_score_soloviev(self, text, func="fre"):
        syllables = len(
            re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", " ".join(text))
        )
        words = len(self.tokenize("".join(text)))
        sentences = len(nltk.sent_tokenize("".join(text)))
        if func == "fre":
            return 208.7 - 2.6 * (words / sentences) - 39.2 * (syllables / words)
        elif func == "fkd":
            return (0.36 * words / sentences) + (5.76 * syllables / words) - 11.97
        else:
            raise ValueError(
                f"Current function is '{func}' but should be in ('fre', 'fkd')"
            )

    def extract_all(self, text):
        print("avg_word_len -- ", self.avg_word_len(text))
        print("avg_sent_len -- ", self.avg_sent_len(text))
        print("avg_words_per_par -- ", self.avg_words_per_prg(text))

        print("char_freq -- ", self.char_freq(text, ","))
        print("lxcp -- ", self.lexical_complexity(text))
        print("ttr -- ", self.ttr(text))

        print(
            "readability_score_oborneva -- ",
            self.readability_score_oborneva(text, "fkd"),
        )
        print("readability_score_soloviev -- ", self.readability_score_soloviev(text))


et = Extractor()
et_t = et.get_text(
    "/Users/apotekhin/repositories/literary-predictor/src/corpus txt/Временщик. Вратарь (СИ) - Дмитрий Билик.txt"
)
print(et.extract_all(et_t))
#  dont forget to cut the edges
