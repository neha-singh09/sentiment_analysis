from gensim.models import Word2Vec
from constants import *

class Embedding_:
    def __init__(self):
        pass

    def create_w2v_model(self, documents):
        w2v_model = Word2Vec(documents, size=W2V_SIZE, window=W2V_WINDOW, 
                    min_count=W2V_MIN_COUNT, workers=W2V_WORKERS)
        # words = list(w2v_model.wv.vocab)
        w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
        return w2v_model
        

