from sklearn.pipeline import Pipeline
import transformers as tfr

variables = ['text']

pipeline = Pipeline([
        ("remove_urls", tfr.RemoveUrls(variables)),
        ("remove_special_characters", tfr.RemoveSpecialCharacters(variables)),
        ("remove_punctuation", tfr.RemovePunctuation(variables)),
        ("tokenize_and_remove_stopwords", tfr.TokenizeRemoveStopWords(variables)),
        ("lemmatization", tfr.Lemmatizer(variables))
])
