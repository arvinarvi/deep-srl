import numpy as np


def get_word_embedding(emb_path, words):
    word_index = {}
    word_embeddings = []
    embeddings = open(emb_path, encoding="utf-8")

    for line in embeddings:
        split = line.strip().split(" ")

        if len(word_index) == 0:  # Add padding+unknown
            word_index["PADDING_TOKEN"] = len(word_index)
            vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
            word_embeddings.append(vector)

            word_index["UNKNOWN_TOKEN"] = len(word_index)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            word_embeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            word_embeddings.append(vector)
            word_index[split[0]] = len(word_index)

    word_embeddings = np.array(word_embeddings)
    return word_index, word_embeddings


def get_char_index_matrix():
    char_index = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char_index[c] = len(char_index)
    return char_index


def get_label_index_matrix():
    label_index = {'A0': 0, 'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'AM-LOC': 5, 'AM-MNR': 6, 'AM-MOD': 7,
                   'AM-NEG': 8, 'AM-TMP':9, '_':10}
    return label_index


def get_pos_index_matrix(POS_tag_set):
    POS_tag_index = {"PAD":0}
    for POS_tag in POS_tag_set:
        POS_tag_index[POS_tag] = len(POS_tag_index)
    return POS_tag_index

def get_deprel_index_matrix(dep_set):
    dep_index = {"PAD":0}
    for dep_tag in dep_set:
        dep_index[dep_tag] = len(dep_index)
    return dep_index

def get_ner_index_matrix(ner_set):
    ner_index = {"PAD":0}
    for ner_tag in ner_set:
        ner_index[ner_tag] = len(ner_index)
    return ner_index
