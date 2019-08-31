# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:35:27 2019

@author: Oliver Huang
"""

import pickle
from keras.models import load_model
from keras import preprocessing
import numpy as np
#import keras.backend as K
#
#K.set_floatx('float16')

input_name = "Query: "
Responder_name = 'Answer:'
MAX_LEN = 10

print('Initializing...')

# load the tokenizer you created in model building
with open('', 'rb') as f:
    input_tokenizer = pickle.load(f)

# load keras model 
model = load_model('')

model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

print('For exit, type "quit"')

exit_code = ['quit', 'exit']


inp = ''

while inp not in exit_code:
    inp = input(input_name).strip()
    out_dict = inp.split(' ')
    inp_phrase = input_tokenizer.texts_to_sequences([inp])
    word_wise_sequence = np.concatenate(inp_phrase)
    x_len = [len(message) for message in inp_phrase]
    
    x = preprocessing.sequence.pad_sequences(inp_phrase,
                                             maxlen=MAX_LEN)
    x_rep= []
    for i, rep in enumerate(x_len):
        for _ in range(rep):
            x_rep.append(x[i])

    x_rep = np.concatenate(x_rep)
    x_rep = np.reshape(x_rep, (-1,MAX_LEN))
    word_wise_sequence = np.reshape(word_wise_sequence, (-1,1))
    word_wise_sequence = preprocessing.sequence.pad_sequences(word_wise_sequence,
                                                              maxlen=MAX_LEN)
    
    result = model.predict(x=[word_wise_sequence, x_rep])
    position = np.argmax(result, axis=1).tolist()

    answer_dict = {k:v for k,v in zip(out_dict, position)}

    # find the duplicated position. If two words occupy same position
    # find softmax activation and rank from that
    sort_p = np.sort(position)
    duplicate_pos = sort_p[:-1][sort_p[1:]==sort_p[:-1]]
    # duplicate_pos is an array containing duplicated positions
    for dup in duplicate_pos:
        location = position == dup
        for pos, val in enumerate(location):
            if val:
                # get position if its true for duplication
                word = out_dict[pos]
                answer_dict[word] = (1-result[pos].max())+answer_dict[word]

    output_word_list = sorted(answer_dict, key=answer_dict.get)
    output = ' '.join(output_word_list)
    print(Responder_name, output)