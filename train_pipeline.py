from pipeline import pipeline
from data_loading import getData
from constants import TRANS_DATA_FILEPATH
import os.path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from word_embeddings import Embedding_
from neural_network import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def run_pipeline():
    
    if os.path.exists(TRANS_DATA_FILEPATH):
        return pd.read_csv(TRANS_DATA_FILEPATH, header=1)
    else:
        df = getData()
        df_transformed = pipeline.fit_transform(df)
        df_transformed.to_csv(TRANS_DATA_FILEPATH, index=False)
        return df_transformed

def label_encoder(target):
    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(target)
    return integer_encoded

# def one_hot_encoder(target):
#     onehot_encoder = OneHotEncoder(sparse=False)
#     integer_encoded = label_encoder(target)
#     integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
#     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#     return onehot_encoded

def data_processing():
    print('pipeline running')
    train_df = run_pipeline()
    train_df['text'] = train_df['text'].fillna("")
    X_train, y_train = train_df['text'], train_df['target']
    print("label encoder running")
    y_train = label_encoder(y_train)
    return X_train, y_train

def train_network():
    X_train, y_train = data_processing()
    
    embedding = Embedding_()
    print("inside embedding")
    w2v_model = embedding.create_w2v_model([line.split() for line in X_train.values])
    w2v_model.save('w2v_model.bin')

    vocab_size, tokenizer = tokenize(X_train.values)
    X_train = pad_sequences_(X_train, tokenizer)
    
    embedding_matrix = embed_matrix(w2v_model, vocab_size, tokenizer)
    embedding_layer = embed_layer(vocab_size, embedding_matrix)
    print('embedding layer created')
    model = network(embedding_layer)
    print("model training started")
    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0), EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split= 0.1, verbose=1, callbacks=callbacks)
    model.save('twitter_sentiment_model.h5')
    print("training completed")
    return model
    

if __name__=='__main__':
    train_network()