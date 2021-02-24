import re
import string
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load('en_core_web_sm')
nlp.disable_pipes('tagger', 'parser')
global c
c=0

class RemoveUrls(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X):
        return self

    def remove_urls(self, row):
        """This function removes all urls from given row"""
        row = re.sub(r"https?:\/\/\S+",'',row)
        return row

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_urls)
        return X

class RemoveSpecialCharacters(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X):
        return self

    def remove_special_char(self, row):
        """This function removes all special characters from given row"""
        row = re.sub(r'[\[\]@_!#$%^&*()<>?/\|}{~:]','',row)
        return row

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_special_char)
        return X

class RemovePunctuation(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X):
        return self

    def remove_punct(self, row):
        """This function removes punctuations from given row"""
        table = row.maketrans('','',string.punctuation)
        return row.translate(table)

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_punct)
        return X

class TokenizeRemoveStopWords(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X):
        return self

    def remove_stopwords(self, row):
        """This function removes punctuations from given row"""
        doc = nlp(row)
        tokens_without_sw = [word.text for word in doc if not word.is_stop]
        global c
        c+=1
        print(c)
        return " ".join(tokens_without_sw)

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.remove_stopwords)
        return X

class Lemmatizer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X):
        return self

    def lemmatize(self, row):
        doc = nlp(row)
        lemma = [word.lemma_ for word in doc]
        global c
        c+=1
        print(c)
        return " ".join(lemma)

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.lemmatize)
        return X




    



