from pipeline import pipeline
from tensorflow import keras
import pandas as pd
from neural_network import tokenize, pad_sequences_
from train_pipeline import train_network

def predict_pipeline(data):
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, list):
        data = pd.DataFrame(data, columns=['text'])
    elif isinstance(data, str):
        data = [data]
        data = pd.DataFrame(data, columns=['text'])

    df_transformed = pipeline.fit_transform(data)
    X_test = df_transformed['text']
    vocab_size, tokenizer = tokenize(X_test.values)
    X_test = pad_sequences_(X_test, tokenizer)
    # model = keras.models.load_model('./twitter_sentiment_model.h5')
    model = train_network()
    score = model.predict(X_test)
    print(score)

if __name__=='__main__':
    data = input()
    predict_pipeline(data)



