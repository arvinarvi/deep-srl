import sys
path = '/home/aravindh/Documents/PHD/NUS/PGM/Codes/deep-srl'

if path not in sys.path:
    sys.path.append(path)

from src.preprocess import readfile, sent2features, add_chars, create_matrices, create_batches, get_words_labels, transform
from embedding.embedding import get_word_embedding, get_char_index_matrix, get_label_index_matrix, \
    get_pos_index_matrix, get_deprel_index_matrix, get_ner_index_matrix
from src.model.model import get_model
from src.validation import Metrics
from keras.utils import Progbar
import numpy as np
from sklearn import metrics
from itertools import chain
from src.validation import compute_f1
from src.analysis import print_wrong_tags
import json
import pickle
import time


with open('../config.json') as json_data_file:
    config = json.load(json_data_file)

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, pos_tag, head, dep, ner, label = data
        input, output = transform([[tokens, casing, char, pos_tag, head, dep, ner, label]], max(2,len(label)), pos_tag_index, dep_index, ner_index)
        pred = model.predict(input, verbose=False)
#        pred = np.add(np.squeeze(pred[0]), np.flip(np.squeeze(pred[1]), axis=0))
        pred = pred.argmax(axis=-1)  # Predict the classes
        
        output = np.squeeze(output)
        output = np.argmax(output, axis=1)            
        correctLabels.append(output)
        
        pred = np.squeeze(pred)
        predLabels.append(pred)
        b.update(i)

    print(metrics.classification_report(list(chain.from_iterable(correctLabels)), list(chain.from_iterable(predLabels))))
    return predLabels, correctLabels


train = readfile(config['train_extended_file_path']) if config['train_with_validation'] else readfile(config['train_file_path'])
val = readfile(config['valid_file_path'])
test = readfile(config['test_file_path'])
#  
#train = train[0:100]  
#val = val[0:100]
#test = test[0:100]

p = 1

train = [sent2features(s,p) for s in train]
val = [sent2features(s,p) for s in val]
test = [sent2features(s,p) for s in test]

#validation_data = validation

#with open('../data/train.pkl', 'rb') as f:
#    train = pickle.load(f)
#    
#with open('../data/val.pkl', 'rb') as f:
#    val = pickle.load(f)
#    
#with open('../data/test.pkl', 'rb') as f:
#    test = pickle.load(f)

t = time.time()
train = add_chars(train)
val = add_chars(val)
test = add_chars(test)

words, pos_tag_set, deprel_set, ner_set, labelSet = get_words_labels(train, val, test)
label_index = get_label_index_matrix()
pos_tag_index = get_pos_index_matrix(pos_tag_set)
dep_index = get_deprel_index_matrix(deprel_set)
ner_index = get_ner_index_matrix(ner_set)
word_index, wordEmbeddings = get_word_embedding(config['embedding_file_path'],words)
char_index = get_char_index_matrix()

train_set = create_matrices(train, word_index,  label_index, char_index, pos_tag_index, dep_index, ner_index)
validation_set = create_matrices(val, word_index, label_index, char_index, pos_tag_index, dep_index, ner_index)
test_set = create_matrices(test, word_index, label_index, char_index, pos_tag_index, dep_index, ner_index)

batch_size = config['batch_size']
model = get_model(wordEmbeddings, char_index, pos_tag_index, dep_index, ner_index, config)

train_steps, train_batches = create_batches(train_set, batch_size, pos_tag_index, dep_index, ner_index)

idx2Label = {v: k for k, v in label_index.items()}

metric = Metrics(validation_set, idx2Label, pos_tag_index, dep_index, ner_index)

epochs = config['epochs']

if config['train_with_validation']:
    model.fit_generator(generator=train_batches, steps_per_epoch=train_steps, epochs=epochs, verbose=config['training_verbose'])
else:
    model.fit_generator(generator=train_batches, steps_per_epoch=train_steps, epochs=epochs, callbacks=[metric], 
                        verbose=config['training_verbose'])

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_set)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_test, rec_test, f1_test))

print(time.time() - t)
#if config['print_wrong_tags']:
#    predLabels, correctLabels = tag_dataset(validation_set)
#    print_wrong_tags(validation_data, predLabels, idx2Label)
