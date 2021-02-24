#DIRECTORY
TRAIN_DATA_FILEPATH = r'C:\Users\nehas\Desktop\web app_NLP model\data\testdata.csv'
TRANS_DATA_FILEPATH = r'C:\Users\nehas\Desktop\web app_NLP model\data\test_df_transformed.csv'
# TEST_DATA_FILEPATH = r'C:\Users\nehas\Desktop\web app_NLP model\data\testdata.csv'

#DATASET
DATASET_COLUMNS = ['target','ids','date','query','user','text']
DATASET_ENCODING = 'ISO-8859-1'

#WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_MIN_COUNT = 10
W2V_WORKERS = 8
W2V_EPOCH = 32

#KERAS
SEQUENCE_LENGTH = 300
BATCH_SIZE = 20
EPOCHS = 10


if __name__=='__main__':
    pass