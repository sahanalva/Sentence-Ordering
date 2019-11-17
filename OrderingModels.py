import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, Bidirectional, Dropout, BatchNormalization
from keras import Model
from PointerLSTM import PointerLSTM
import keras.backend as K

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
    	signature="default", as_dict=True)["default"]

def PointerLstmBased(max_num_sent):
    sent_seq = Input(shape=(max_num_sent, 1), name='sent_inp', dtype=tf.string)
    sent_emb = TimeDistributed(Lambda(UniversalEmbedding, output_shape=(512,)))(sent_seq)
    sent_lstm = LSTM(100, return_sequences=True)(sent_emb)
    sent_lstm = Dropout(0.5)(sent_lstm)
    pointer_lstm = PointerLSTM(100, units=100)(sent_lstm)
    model = Model(inputs = sent_seq, outputs=pointer_lstm)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def BiLstmBased(max_num_sent):
    sent_seq = Input(shape=(max_num_sent, 1), name='sent_inp', dtype=tf.string)
    sent_emb = TimeDistributed(Lambda(UniversalEmbedding, output_shape=(512,)))(sent_seq)
    sent_lstm = Bidirectional(LSTM(100, return_sequences=True))(sent_emb)
    sent_lstm = Dropout(0.5)(sent_lstm)
    final_lstm = Bidirectional(LSTM(300, return_sequences=True))(sent_lstm)
    final_lstm = Dropout(0.5)(final_lstm)
    output_seq = TimeDistributed(Dense(max_num_sent + 1, activation='softmax'))(final_lstm) 
    model = Model(inputs = sent_seq, outputs=output_seq)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def BiLstmBasedWordEmbed(max_num_sent, max_sent_len, embedding_layer):
    word_seq = Input(shape=(max_num_sent, max_sent_len), dtype = 'int32')
    word_emb = TimeDistributed(embedding_layer)(word_seq)
    sent_emb = TimeDistributed(Bidirectional(LSTM(300)))(word_emb)
    sent_lstm = Bidirectional(LSTM(200, return_sequences=True))(sent_emb)
    sent_lstm = Dropout(0.5)(sent_lstm)
    final_lstm = Bidirectional(LSTM(300, return_sequences=True))(sent_lstm)
    final_lstm = Dropout(0.5)(final_lstm)
    output_seq = TimeDistributed(Dense(max_num_sent + 1, activation='softmax'))(final_lstm) 
    model = Model(inputs = word_seq, outputs=output_seq)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model