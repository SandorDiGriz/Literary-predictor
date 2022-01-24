from pymorphy2 import MorphAnalyzer
from functools import lru_cache
from tqdm import tqdm

import nltk
import numpy as np
import re
import math


class Extractor:
    def get_text(self, path):
        with open(path, "r", encoding="utf-8") as file:
            text = file.readlines()

        return text

    def tokenize(self, text, regex="[А-Яа-яёA-z]+"):
        regex = re.compile(regex)
        tokens = regex.findall(text.lower())

        return tokens

    def remove_stopwords(lemmas, stopwords=nltk.corpus.stopwords.words("russian")):
        return [w for w in lemmas if not w in stopwords and len(w) > 3]

    @lru_cache(maxsize=128)
    def lemmatize_word(self, token):
        pymorphy = MorphAnalyzer()
        return pymorphy.parse(token)[0].normal_form

    def lemmatize_text(self, text):
        return [self.lemmatize_word(w) for w in tqdm(text)]

    def avg_sent_len(self, text, cut_edge=False):
        sentences = nltk.sent_tokenize("".join(text))
        sent_len = [len(sent.split()) for sent in sentences]
        if len(sent_len) > 20 and cut_edge:
            sent_len = sent_len[20:]

        return np.mean(sent_len)

    def avg_word_len(self, text, cut_edge=False):
        tokens = set(self.tokenize("".join(text)))
        word_len = [len(token) for token in tokens]
        if len(word_len) > 500 and cut_edge:
            word_len = word_len[500:]

        return np.mean(word_len)

    def char_freq(text, char):
        tokens = nltk.word_tokenize("".join(text))
        char_distfreq = nltk.probability.FreqDist(tokens)

        return (char_distfreq[str(char)] * 1000) / char_distfreq.N()

    def lexical_complexity():
        pass

    def ttr(self, text, mode="standard"):
        tokens = self.tokenize("".join(text))
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
        syllables = len(re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", "".join(text)))
        words = len(self.tokenize("".join(text)))
        if func == "fre":
            return 206.835 - 1.3 * (words / syllables) - 60.1 * (syllables / words)
        elif func == "fkd":
            return (0.5 * words / syllables) + (8.4 * syllables / words) - 15.59
        else:
            raise ValueError(
                f"Current function is '{func}' but should be in ('fre', 'fkd')"
            )

    def readability_score_soloviev(self, text, func="fre"):
        syllables = len(re.findall("[АаУуОоЫыИиЭэЯяЮюЁёЕеAaEeIiOoUuYy]", "".join(text)))
        words = len(self.tokenize("".join(text)))
        if func == "fre":
            return 208.7 - 2.6 * (words / syllables) - 39.2 * (syllables / words)
        elif func == "fkd":
            return (0.36 * words / syllables) + (5.76 * syllables / words) - 11.97
        else:
            raise ValueError(
                f"Current function is '{func}' but should be in ('fre', 'fkd')"
            )


et = Extractor()
et_t = et.get_text(
    "/Users/apotekhin/repositories/literary-predictor/src/corpus txt/«Я понял жизни цель» (проза, стихотворения, поэмы, переводы) - Борис Пастернак.txt"
)
print(et.ttr(et_t))
