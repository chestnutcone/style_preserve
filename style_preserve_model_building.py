# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:42:12 2019

@author: Oliver Huang


"""

import pickle
import matplotlib.pyplot as plt
from keras import preprocessing, models, layers
import numpy as np
import copy

# input was from facebook convo formmated into list of msgs in str
# in format ['..','...','...',....]
# laod your data here
with open('', 'rb') as f:
    flatten_input = pickle.load(f)


MAX_WORDS = 10000 # for input tokenizer. Only top 10k words will be used
MAX_LEN = 10 # for input tokenizer
MAX_ORDER = 10 # the maximum order the model can rearrange

input_tokenizer = preprocessing.text.Tokenizer(num_words=MAX_WORDS,
                                               oov_token='UNK')
input_tokenizer.fit_on_texts(flatten_input)
clean_input_sequences = input_tokenizer.texts_to_sequences(flatten_input)
input_word_index = input_tokenizer.word_index


# create training label
# the softmax will output probability of which position the word is most likely
# to occupy. Create using identity matrix since everything is in order to start
y = [np.identity(len(sentence)) for sentence in clean_input_sequences]

# noisy input sequence is shuffled clean_input_sequence
noisy_input_sequences = copy.deepcopy(clean_input_sequences)
for i in range(len(noisy_input_sequences)):
    data_size = len(noisy_input_sequences[i])
#
    # start shuffling orders of words within a message
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    y[i] = y[i][indices]
    noisy_input_sequences[i] = np.asarray(noisy_input_sequences[i])[indices]

    # pad to MAX_ORDER
    # this may cause issue about rearrange (lose information on rearrangements)
    # since it crops off after shuffle, but majority will be fine
    if data_size < MAX_ORDER:
        temp = np.zeros((data_size, MAX_ORDER))
        temp[:y[i].shape[0], :y[i].shape[1]] = y[i]
#        y[i] = np.vstack((temp, np.ones((1,MAX_ORDER))))
        y[i] = temp
    else:
        # when it is array is bigger than max_order, need to crop
        y[i] = y[i][:data_size, :MAX_ORDER]

y = np.concatenate(y)  # flatten it

# word_wise_sequence will feed into neural net word by word
word_wise_sequence = np.concatenate(noisy_input_sequences)
x_len = [len(message) for message in noisy_input_sequences]

x = preprocessing.sequence.pad_sequences(noisy_input_sequences,
                                         maxlen=MAX_LEN)
# x_rep will repeat the whole phrase for # of words
# say phrase A has 5 words. word_wise_sequence will feed 5 words individually
# while x_rep will feed phrase A 5 times
x_rep= []
for i, rep in enumerate(x_len):
    for _ in range(rep):
        x_rep.append(x[i])

x_rep = np.concatenate(x_rep)
x_rep = np.reshape(x_rep, (-1,MAX_LEN))

word_wise_sequence = np.reshape(word_wise_sequence, (-1,1))
word_wise_sequence = preprocessing.sequence.pad_sequences(word_wise_sequence,
                                                          maxlen=MAX_LEN)

# split data
split_ratio = 0.8
split_len = int(len(y)*split_ratio)
x_rep_train = x_rep[:split_len]
x_rep_val = x_rep[split_len:]

word_train = word_wise_sequence[:split_len]
word_val = word_wise_sequence[split_len:]

y_train = y[:split_len]
y_val = y[split_len:]


def build_model(softmax_output, maxlen, vocab_size):
    inputs = layers.Input(shape=(maxlen,), name='whole_phrase')
    previous_output = layers.Input(shape=(maxlen,), name='word_wise')

    embed = layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)
    central = embed(inputs)
    pre_out = embed(previous_output)
    central = layers.concatenate([central, pre_out], axis=1)

    central = layers.Flatten()(central)
    central = layers.Dense(16, activation='relu')(central)
    central = layers.Dense(16, activation='relu')(central)
    central = layers.Dense(16, activation='relu')(central)

    central = layers.Dense(softmax_output, activation='softmax')(central)
    central_model = models.Model(inputs=[inputs, previous_output], outputs=central)
    central_model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    return central_model

def plot_graph(all_history):
    """
    This will plot data from the output of fit method
    """
    acc = all_history['acc']
    val_acc = all_history['val_acc']
    loss = all_history['loss']
    val_loss = all_history['val_loss']
    run_length = range(1, len(acc)+1)
    plt.plot(run_length, acc, label='acc')
    plt.plot(run_length, val_acc, label='val acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()

    plt.figure()
    plt.plot(run_length, loss, label='loss')
    plt.plot(run_length, val_loss, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


model = build_model(softmax_output=MAX_ORDER,
                    vocab_size=MAX_WORDS,
                    maxlen=MAX_LEN)

history = model.fit(x=[word_train, x_rep_train], y=y_train, epochs=10,
                    batch_size=512, validation_data=([word_val, x_rep_val],y_val))

plot_graph(history.history)
