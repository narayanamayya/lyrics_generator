import os
import json
import argparse
import nltk
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from model import build_model, save_weights
from tensorflow.keras.preprocessing.text import Tokenizer
DATA_DIR = '/home/narayana.mayya@sapt.local/projects/Antoine_coefficient_modelling/extra_stuff/Test/data'
BATCH_SIZE = 16
SEQ_LENGTH = 64
from keras.models import Sequential, load_model


def read_batches(T, vocab_size):
    length = T.shape[0] #448858
    batch_chars = int(length / BATCH_SIZE) #28053 
    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): 
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 16X64
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) 
        for batch_idx in range(0, BATCH_SIZE): # (0,16)
            for i in range(0, SEQ_LENGTH): #(0,64)
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y
def train(epochs=100, save_freq=10):
    # Load text data from file
    text_data_file_path = '/home/narayana.mayya@sapt.local/projects/Antoine_coefficient_modelling/extra_stuff/Test/data/input.txt'
    with open(text_data_file_path, 'r') as file:
        text_data = file.read().splitlines()
    # Initialize tokenizer
    tokenizer = Tokenizer()
    # Fit tokenizer on text data
    tokenizer.fit_on_texts(text_data)
    '''
    tokenizer_file_path = '/home/narayana.mayya@sapt.local/projects/Antoine_coefficient_modelling/extra_stuff/Test/model/tokenizer_if_needed.pkl'
    with open(tokenizer_file_path, 'wb') as file:
        pickle.dump(tokenizer, file)
    '''
    # Create a dictionary of unique words
    word_index = tokenizer.word_index
    # Print the dictionary
    #print("All unique words from raw data: ", len(word_index)) #8130 
    # Download the English word corpus from NLTK
    nltk.download('words')
    # Get the set of English words
    english_words = set(nltk.corpus.words.words())
    # Filter out words in word_index that are not valid English words
    valid_word_index = {word:index for word,index in word_index.items() if word in english_words}
    char_to_idx = {key: index + 1 for index, key in enumerate(valid_word_index)}   
    #char_to_idx = valid_word_index 

    #print("Number of unique characters: " + str(len(char_to_idx))) #4474

    with open(os.path.join(DATA_DIR, 'char_to_idx_dictcheck_indexreset_run3.json'), 'w') as f:
        json.dump(char_to_idx, f)

    #idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx) #4474

    #model_architecture
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model = load_model("/home/narayana.mayya@sapt.local/projects/Antoine_coefficient_modelling/extra_stuff/Test/model/model_run4.h5") 
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train data generation
    text = ' '.join(text_data)
    T = tokenizer.texts_to_sequences(text)
    #print("Length of text:" + str(T.size)) 
    T = np.asarray([lst for lst in T if lst], dtype=np.int32)
    #print(T.shape) #(506795, 1)
    mask = np.all(np.isin(T, list(char_to_idx.values())), axis=1)
    T = T[mask] 
    #print(T.shape) #(448858, 1)
    #steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))        
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):            
            #print(X);
            loss, acc = model.train_on_batch(X, Y)
        print('loss = {}, acc = {}'.format(loss, acc))
        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights_test.{}.h5'.format(epoch + 1))
    model.save('/home/narayana.mayya@sapt.local/projects/Antoine_coefficient_modelling/extra_stuff/Test/model/model_test.h5')

train(epochs=2, save_freq=10)
