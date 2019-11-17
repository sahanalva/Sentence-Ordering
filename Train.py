import numpy as np
from OrderingModels import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Model
from keras.initializers import Constant
import pickle

data_folder = '.\\data\\processed\\' # Fill out locations of sentences and permutations file created by data preparation notebooks
split_at = 185000
batch_size = 100
max_seq_len = 10
max_sent_len = 40
max_num_word = 2000
EMBEDDING_DIM = 300
usePointerBasedLSTM = False # Work in progress
useWordLevelEmbeddings = True
usePretrainedWordEmbeddings = True

y = np.loadtxt(data_folder + 'permutations.txt', delimiter='\t', dtype=int)

with open(data_folder + 'sentences.txt', encoding='utf8') as f:
    sentences = f.readlines()

inc = 1
if(usePointerBasedLSTM):
    # pointer lstm assumes output decision space has equal length to input.
    inc = 0 

YY = []
for y_ in y:
    yy = []
    dummyVec = np.array([ 0 for i in range(max_seq_len + inc)])
    for yy_ in y_:
        dummyVec[yy_] = 1
        yy.append(np.copy(dummyVec))
        dummyVec[yy_] = 0
    YY.append(yy)

YY = np.asarray(YY)

tokenizer_file_extra = ''
if(useWordLevelEmbeddings):

    tokenizer = Tokenizer(num_words=max_num_word)
    tokenizer.fit_on_texts(sentences)
    
    X = []
    for line in sentences:
        lineSents = line.split('\t')
        xx = np.zeros((max_seq_len, max_sent_len))
        j = 0
        for ls in lineSents:
            ls_vec = tokenizer.texts_to_sequences([ls])
            ls_vec = pad_sequences(ls_vec, maxlen=max_sent_len)
            xx[j] = np.copy(ls_vec[0])
            j+= 1
        X.append(np.copy(xx))

    X = np.asarray(X)

    word_index = tokenizer.word_index
    num_words = min(max_num_word, len(word_index) + 1)

    if(usePretrainedWordEmbeddings):
        embeddings_index = {}
        with open('.\\Pretrained\\glove.6B\\glove.6B.300d.txt', encoding="utf8") as f:
            for line in f.readlines():
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= max_num_word:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        embedding_layer = Embedding(num_words, EMBEDDING_DIM, input_length=max_sent_len, embeddings_initializer=Constant(embedding_matrix), trainable=False)
        tokenizer_file_extra = '_pretrained'

    else:
        embedding_layer = Embedding(num_words, EMBEDDING_DIM, input_length=max_sent_len, trainable=True)

else:
    X = []
    for line in sentences:
        lineSents = line.split('\t')
        xx = [[" "] for i in range(max_seq_len)]
        j = 0
        for ls in lineSents:
            xx[j][0] = ls
            j += 1
        X.append(xx)
    
    X = np.asarray(X)
    
# split into train and test data
x_train = X[:split_at]
x_test = X[split_at:]

y_train = YY[:split_at]
y_test = YY[split_at:]

validation_data = (x_test, y_test)

with open('.\\Tokenizer\\tokenizer' + tokenizer_file_extra + '.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if(usePointerBasedLSTM):
    model = PointerLstmBased(max_seq_len)
elif(useWordLevelEmbeddings):
    model = BiLstmBasedWordEmbed(max_seq_len,max_sent_len, embedding_layer)
else:
    model = BiLstmBased(max_seq_len)

with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size,
                    validation_data=validation_data, shuffle=True)

  model.save_weights('.\\Models\\model.h5')