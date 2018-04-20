import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import re, string
import csv

args = {'_', 'A0', 'A1', 'A2', 'A3', 'A4', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-TMP'}

def clearup(s, chars):
    str = re.sub('[%s]' % chars, '0', s)
    str = re.sub('-+', '-', str)
    return str

def readfile(filepath):
    
    with open(filepath, 'r') as f:  #opens PW file
        #reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))
        
    new_data = []
    j=0
    for i in range(0, len(data)):
        new_data.append([])
        #if data[i][1] != '0' and data[i][2] != '0' and data[i][3] != '0': #'.' for enhanced features        
        if(len(data[i]) > 0 and len(data[i][1]) != 0): # For original dataset read
            new_data[j].append(data[i])
        else:
            j = j + 1
        
    n = filter(None, new_data)
    new_data = list(n)    
    return new_data


def word2features(sent, i, p):    
    lemma = sent[i][2]
    postag = sent[i][4]    
    head = sent[int(sent[i][8])-1][1] #Head word index start from 1 in the dataset
    deprel = sent[i][10]
    nerTag = sent[i][14]        
    
    if(sent[i][15 + p - 1] != '' and sent[i][15 + p - 1] in args):
        label = sent[i][15 + p - 1]
    else:            
        label = '_'
            
    features = [lemma, postag, head, deprel, nerTag, label]
        
    return features

def sent2features(sent, p):    
    
    feat = []
    
    for i in range(len(sent)):        
        feat.append(word2features(sent, i, p))
        
    return feat


#def sent2labels(sent, p):
#    label = list()              
#    
#    for l in sent:                              
#        if(l[15 + p - 1] != '' and l[15 + p - 1] in args):
#            label.append(l[15 + p - 1])
#        else:            
#            label.append('_')
#        
#    return label


def add_chars(sentences):
    for sentence_index, sentence in enumerate(sentences):
        for word_index, word_info in enumerate(sentence):
            chars = [c for c in word_info[0]]
            sentences[sentence_index][word_index] = [word_info[0], chars, word_info[1], word_info[2], word_info[3], word_info[4], word_info[5]]
    return sentences


def get_casing(word):
    casing = []

    num_of_digits = 0
    for char in word:
        if char.isdigit():
            num_of_digits += 1

    num_of_digits_norm = num_of_digits / float(len(word))

    casing.append(1) if word.isdigit() else casing.append(0)
    casing.append(1) if num_of_digits_norm > 0.5 else casing.append(0)
    casing.append(1) if word.islower() else casing.append(0)
    casing.append(1) if word.isupper() else casing.append(0)
    casing.append(1) if word[0].isupper() else casing.append(0)
    casing.append(1) if num_of_digits > 0 else casing.append(0)
    casing.append(1) if word.isalnum() > 0 else casing.append(0)
    casing.append(1) if word.isalpha() > 0 else casing.append(0)
    casing.append(1) if word.find("\'") >= 0 else casing.append(0)
    casing.append(1) if word == "(" or word == ")" else casing.append(0)
    casing.append(1) if len(word) == 1 else casing.append(0)

    return casing


def create_matrices(sentences, word_index, label_index, char_index, pos_tag_index, dep_index, ner_index):
    unknown_index = word_index['UNKNOWN_TOKEN']
    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []
        pos_tag_indices = []
        dep_indices = []
        ner_indices = []

        for word, char, pos_tag, head, deprel, ner, label in sentence:            
            word_count += 1
            if word in word_index:
                word_idx = word_index[word]
            elif word.lower() in word_index:
                word_idx = word_index[word.lower()]
            else:
                word_idx = unknown_index
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char_index[x])
            # Get the label and map to int
            word_indices.append(word_idx)
            case_indices.append(get_casing(word))
            char_indices.append(char_idx)
            label_indices.append(label_index[label])
            pos_tag_indices.append(pos_tag_index[pos_tag])
            dep_indices.append(dep_index[deprel])
            ner_indices.append(ner_index[ner])

        dataset.append([word_indices, case_indices, char_indices, pos_tag_indices, dep_indices, ner_indices, label_indices])

    return dataset


def padding(chars, length):
    padded_chair = []
    for i in chars:
        padded_chair.append(pad_sequences(i, length, padding='post'))
    return padded_chair


def create_batches(data, batch_size, pos_tag_index, dep_index, ner_index):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        def get_length(data):
            word, case, char, pos_tag, dep, ner, label = data
            return len(word)

        data_size = len(data)
        data.sort(key=lambda x: get_length(x))

        while True:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = data[start_index: end_index]
                max_length_word = max(len(max(seq, key=len)) for seq in X)
                yield transform(X, max(2,max_length_word), pos_tag_index, dep_index, ner_index)

    return num_batches_per_epoch, data_generator()


def transform(X, max_length_word, pos_tag_index, dep_index, ner_index):
    word_input = []
    char_input = []
    case_input = []
    label_input = []
    pos_tag_input = []
    dep_input = []
    ner_input = []

    max_length_char = find_max_length_char(X)

    for word, case, char, pos_tag, dep, ner, label in X:
        word_input.append(pad_sequence(word, max_length_word))
        case_input.append(pad_sequence(case, max_length_word, False, True))
        label_input.append(np.eye(11)[pad_sequence(label, max_length_word)])
        pos_tag_input.append(to_categorical(pad_sequence(pos_tag, max_length_word), num_classes=len(pos_tag_index)))
        dep_input.append(to_categorical(pad_sequence(dep, max_length_word), num_classes=len(dep_index)))
        ner_input.append(to_categorical(pad_sequence(ner, max_length_word), num_classes=len(ner_index)))
        char_input.append(pad_sequence(char, max_length_word, True))
    
    return [np.asarray(word_input), np.asarray(case_input), np.asarray(pos_tag_input), np.asarray(dep_input), np.asarray(ner_input), np.asarray(padding(char_input, max_length_char))], np.asarray(label_input) #np.flip(np.asarray(label_input), axis=1)]


def find_max_length_char(X):
    max_length = 0;
    for word, case, char, pos_tag, dep, ner, label in X:
        for ch in char:
            if len(ch) > max_length:
                max_length = len(ch)
    return max_length


def pad_sequence(seq, pad_length, isChair = False, isCasing = False):
    if isChair:
        for x in range(len(seq), pad_length):
            seq.append([])
        return seq
    elif isCasing:
        for x in range(pad_length - len(seq)):
            seq.append(np.zeros(11))
        return seq
    else:
        return np.pad(seq, (0, pad_length - len(seq)), 'constant', constant_values=(0,0))


def get_words_labels(train, val, test):    
    pos_tag_set = set()
    label_set = set()
    deprel_set = set()
    ner_set = set()
    words = {}

    for dataset in [train, val, test]:
        for sentence in dataset:
            for word, char, POS_tag, head, deprel, ner, label in sentence:                
                pos_tag_set.add(POS_tag)
                label_set.add(label)
                deprel_set.add(deprel)
                ner_set.add(ner)
                words[word.lower()] = True
    return words, pos_tag_set, deprel_set, ner_set, label_set