########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import six.moves.cPickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
########################################
## set directories and parameters
########################################
BASE_DIR = './'
EMBEDDING_FILE = '/root/qa/data/GoogleNews-vectors-negative300.bin'
# TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
# TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

# num_lstm =120
# num_dense = 40
# rate_drop_lstm =.25
# rate_drop_dense =.25

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

STAMP_MME = 'lstm_mme_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)
########################################
## index word vectors
########################################
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

########################################
## process texts in datasets
########################################
print('Processing text dataset')


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()
    #     print( 'text  1 ',text)
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # text = text.lower().split()
    # Return a list of words
    return (text)

############prepare  training/validation data  /only use mme
train_processed_query_dataset_path ='/root/qa/data/faq_query_pairs609.csv'
train_processed_query_df = pd.read_csv(train_processed_query_dataset_path)
print("processed train data set shape",train_processed_query_df.shape)
train_texts_mme_1 = list(train_processed_query_df.loc[:,'query'].map(lambda x: text_to_wordlist(x,remove_stopwords=True, stem_words=True) ))
train_texts_mme_2 = list(train_processed_query_df.loc[:,'faq'].map(lambda x: text_to_wordlist(x,remove_stopwords=True, stem_words=True) ))
train_labels_mme   = list(train_processed_query_df.loc[:,'match'])
###########add the pair of (query1,query1)  sentence as positive samples
new_positive = list(set(train_texts_mme_1))+ list(set(train_texts_mme_2))
print('the number of set of faq and query as positive ',len(new_positive))
train_texts_mme_add_1 = train_texts_mme_1 + new_positive
train_texts_mme_add_2 = train_texts_mme_2 + new_positive
train_labels_mme_add  = train_labels_mme  + [1]* len(new_positive)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_texts_mme_1 + train_texts_mme_2 )
tokenizer_fname = 'tokenizer1'
six.moves.cPickle.dump(tokenizer, open(os.path.join('./result/', tokenizer_fname), "wb"))
# tokenizer = six.moves.cPickle.load(open(os.path.join('./model/', tokenizer_fname), 'rb'))
train_sequences_mme_add_1 = tokenizer.texts_to_sequences(train_texts_mme_add_1)
train_sequences_mme_add_2 = tokenizer.texts_to_sequences(train_texts_mme_add_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
train_data_mme_add_1 = pad_sequences(train_sequences_mme_add_1, maxlen=MAX_SEQUENCE_LENGTH)
train_data_mme_add_2 = pad_sequences(train_sequences_mme_add_2, maxlen=MAX_SEQUENCE_LENGTH)
train_labels_mme_add = np.array(train_labels_mme_add)
print('Shape of train_data tensor:', train_data_mme_add_1.shape)
print('Shape of train_label tensor:', train_labels_mme_add.shape)


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

tokenizer = six.moves.cPickle.load(open(os.path.join('./result/', tokenizer_fname), 'rb'))


word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))+1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
#         if word =='import':
#             print(embedding_matrix[i])
#             2.06054688e-01   2.27539062e-01  -2.13867188e-01   6.49414062e-02
#    4.12597656e-02   8.49609375e-02   1.48437500e-01  -2.91015625e-01
#   -7.08007812e-02   2.02148438e-01  -1.56402588e-03  -1.19140625e-01
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
embedding_matrix_path = './result/embedding_matrix.npy'
np.save(embedding_matrix_path,embedding_matrix)

# ######################################
# sample train/validation data
# #######################################
np.random.seed(1234)
perm_mme_add = np.random.permutation(len(train_data_mme_add_1))
idx_train_mme_add = perm_mme_add[:int(len(train_data_mme_add_1)*(1-VALIDATION_SPLIT))]
idx_val_mme_add =  perm_mme_add[int(len(train_data_mme_add_1)*(1-VALIDATION_SPLIT)):]
print('idx_train len', len(idx_train_mme_add))
print('idx_val  len',len(idx_val_mme_add))

data_train_mme_add_1  = np.vstack((train_data_mme_add_1[idx_train_mme_add], train_data_mme_add_2[idx_train_mme_add]))
data_train_mme_add_2  = np.vstack((train_data_mme_add_2[idx_train_mme_add], train_data_mme_add_1[idx_train_mme_add]))
labels_train_mme_add  = np.concatenate((train_labels_mme_add[idx_train_mme_add], train_labels_mme_add[idx_train_mme_add]))
print("data_train_mme_add_1 lenth", data_train_mme_add_1.shape[0])
data_val_mme_add_1  = np.vstack((train_data_mme_add_1[idx_val_mme_add], train_data_mme_add_2[idx_val_mme_add]))
data_val_mme_add_2  = np.vstack((train_data_mme_add_2[idx_val_mme_add], train_data_mme_add_1[idx_val_mme_add]))
labels_val_mme_add  = np.concatenate((train_labels_mme_add[idx_val_mme_add], train_labels_mme_add[idx_val_mme_add]))
print("labels_val_mme_add length", labels_val_mme_add.shape[0])
weight_val = np.ones(len(labels_val_mme_add))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val_mme_add==0] = 1.309028344

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)
########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

# t2 = time.time()
########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)

model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
# model.summary()jia
# weight_val = np.ones(len(labels_val_mme_add))
# if re_weight:
#     weight_val *= 0.472001959
#     weight_val[labels_val_mme_add==0] = 1.309028344
# STAMP_MME = 'lstm_mme_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
#         rate_drop_dense)
print(STAMP_MME)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
bst_model_path = STAMP_MME + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_train_mme_add_1, data_train_mme_add_2], labels_train_mme_add, \
                 validation_data=([data_val_mme_add_1, data_val_mme_add_2], labels_val_mme_add, weight_val), \
                 epochs=200, batch_size=2048, shuffle=True, \
                 class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
###########predict on the mme data
TEST_DATA_FILE2 = '/root/qa/data/faq_query_pairs74.csv'
pre_processed_query_df = pd.read_csv(TEST_DATA_FILE2)
pre_texts_1 = list(pre_processed_query_df.loc[:,'query'].map(lambda x: text_to_wordlist(x,remove_stopwords=True, stem_words=True) ))
pre_texts_2 = list(pre_processed_query_df.loc[:,'faq'].map(lambda x: text_to_wordlist(x,remove_stopwords=True, stem_words=True) ))

# print(test_texts_1)
pre_sequences_1 = tokenizer.texts_to_sequences(pre_texts_1)
pre_sequences_2 = tokenizer.texts_to_sequences(pre_texts_2)
print(pre_sequences_1[0])


pre_data_1 = pad_sequences(pre_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
pre_data_2 = pad_sequences(pre_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', pre_data_1.shape)
############compute socre for every pair of faq and query
import time
t1 = time.time()
preds1 = model.predict([pre_data_1, pre_data_2], batch_size=8192, verbose=1)
preds2= model.predict([pre_data_2, pre_data_1], batch_size=8192, verbose=1)
preds =(preds1 +preds2)/2
pre_processed_query_df.loc[:,'score'] =preds
pre_processed_query_df.to_csv("./result/result_query_df_train_mme600.csv")

anser_index = pre_processed_query_df.groupby('query').apply(lambda subf: subf.loc[:,'score'].argmax())
anser_index2 = list(anser_index)
currectnum = pre_processed_query_df.loc[list(anser_index),'match'].sum()
t2 = time.time()
print(list(anser_index))
print('currectnum------------------------',currectnum)
print(pre_processed_query_df.loc[list(anser_index),:])
print("time of prediction for each query",(t2-t1)/len(anser_index2))