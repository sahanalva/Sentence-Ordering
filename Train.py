from UniversalEmbeddingModels import *
import numpy as np
from keras.layers import LSTM, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

split_at = 1500
batch_size = 50
max_seq_len = 10

y = np.loadtxt('.\\data\\processed\\permutations.txt', delimiter='\t', dtype=int)

with open('.\\data\\processed\\sentences.txt', encoding='utf8') as f:
    sentences = f.readlines()

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

YY = []
for y_ in y:
    yy = []
    dummyVec = np.array([ 0 for i in range(max_seq_len + 1)])
    for yy_ in y_:
        dummyVec[yy_] = 1
        yy.append(np.copy(dummyVec))
        dummyVec[yy_] = 0
    YY.append(yy)
    
YY = np.asarray(YY)

x_train = X[:split_at]
x_test = X[split_at:]

y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]

validation_data = (x_test, YY_test)

# model = PointerLstmBased(15)
model = BiLstmBased(max_seq_len)

with tf.Session() as session:
  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  history = model.fit(x_train, YY_train, epochs=100, batch_size=batch_size,
                    validation_data=validation_data)
  model.save_weights('.\\Models\\model.h5')




# p = model.predict(x_test)