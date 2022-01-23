import nltk
import numpy as np
import re


class Extractor:
    def get_text(self, path):
        with open(path, "r", encoding="utf-8") as file:
            text = file.readlines()

        return text

    def tokenize(self, text, regex="[А-Яа-яёA-z]+"):
        regex = re.compile(regex)
        tokens = regex.findall(text.lower())

        return tokens

    def avg_sent_len(self, file):
        text = self.get_text(file)
        sentences = nltk.sent_tokenize("".join(text))
        word_len = [len(sent.split()) for sent in sentences]
        if len(word_len) > 20:
            word_len = word_len[20:]

        return np.mean(word_len)

    def avg_word_len(self, file):
        text = self.get_text(file)
        tokens = self.tokenize("".join(text))
        word_len = [len(token) for token in tokens]
        if len(word_len) > 500:
            word_len = word_len[500:]

        return np.mean(word_len)

    def lexical_diversity(self, file):
        text = self.get_text(file)
        tokens = self.tokenize("".join(text))
        return (len(set(tokens)) / len(tokens)) * 100


et = Extractor()
print(
    et.lexical_diversity(
        "/Users/apotekhin/repositories/literary-predictor/src/corpus txt/1test.txt"
    )
)
