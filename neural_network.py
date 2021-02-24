import numpy as np
from constants import *
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def tokenize(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    vocab_size = len(tokenizer.word_index)+1
    return vocab_size, tokenizer

def pad_sequences_(text, tokenizer):
    padded_seq = pad_sequences(tokenizer.texts_to_sequences(text), maxlen= SEQUENCE_LENGTH)
    return padded_seq

def embed_matrix(w2v_model, vocab_size, tokenizer):
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

def embed_layer(vocab_size, embedding_matrix):
    embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], 
                        input_length=SEQUENCE_LENGTH, trainable=False)
    return embedding_layer

def network(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model
