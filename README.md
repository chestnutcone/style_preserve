# style_preserve
A simple neural network designed to preserve style (and therefore grammar) of a person's text response. In short, it takes in scrambled word sequence and outputs a organized sequence. This particular network works about 30% of the time.

Files:\
style_preserve_model_building.py contains codes for building Keras model\
style_preserve_main.py runs the model

How it works:
I downloaded facebook conversation data (JSON) and made it into a list of messages. The neural network has two input: 1. Receive each word one by one 2. Receive the whole list of words. The dense network outputs softmax on the probability of position where a word would reside. If there are multiple words occupying the same position, it will rank by the confidence of softmax probability. 

Sample output from network:

Query: am i having a time wonderful\
Answer: i am having a wonderful time

Query: are you how\
Answer: how are you

Query: your day is how\
Answer: how is your day

Query: to meet you nice\
Answer: you to meet nice

Query: am hungry i\
Answer: i am hungry

Query: go lets to walmart\
Answer: lets go to walmart

Query: a dog is this\
Answer: is a dog this

Query: mcdonalds go to\
Answer: go to mcdonalds

Query: have to i to go mcdonalds\
Answer: i have to mcdonalds go

Query: nice day is a today\
Answer: nice is a today day

