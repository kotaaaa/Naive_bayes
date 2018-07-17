from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class NaiveBayses:
    def __init__(self, file1, file2, mode):
        f = open(file1)
        self.file1_text = f.read()
        f = open(file2)
        self.file2_text = f.read()

        if mode == "jpn":
           self.cv = CountVectorizer(analyzer=self.split_words)
        else:
           self.cv = CountVectorizer()

        self.cv.fit([self.file1_text, self.file2_text])
        self.clf = MultinomialNB()
        self.x1 = self.cv.transform([self.file1_text])
        self.x2 = self.cv.transform([self.file2_text])
        X = np.concatenate((self.x1.toarray(), self.x2.toarray()))
        self.clf.fit(X, [1,2])

    def split_words(self,text):
        t = Tokenizer()
        tokens =  t.tokenize(text)
        return [token.surface for token in tokens]

    def get_words_frequency_file1(self):
        return self.x1.toarray()

    def get_words_frequency_file2(self):
        return self.cv.transform([self.file2_text]).toarray()

    def get_vocabulary(self):
        return self.cv.vocabulary_

    def predict(self, text):
        x = self.cv.transform([text]).toarray()
        return self.clf.predict(x)

    def predict_log_proba(self, text):
        x = self.cv.transform([text]).toarray()
        return self.clf.predict_log_proba(x)

    def predict_proba(self, text):
        x = self.cv.transform([text]).toarray()
        return self.clf.predict_proba(x)
