import pandas as pd
import numpy as np
from constants import *

def getData():
    df = pd.read_csv(TRAIN_DATA_FILEPATH, encoding = DATASET_ENCODING, header = None, names = DATASET_COLUMNS)
    df = df.fillna("")
    usefulColumns = ['target','text']
    df = df[usefulColumns]
    df['target'] = df['target'].astype(np.int8)
    return df

if __name__=='__main__':
    twitter_df = getData()


