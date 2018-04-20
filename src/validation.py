from keras.callbacks import Callback
from src.preprocess import transform
import numpy as np
from sklearn.metrics import precision_score

# Method to compute the accuracy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label):    
    assert(len(predictions) == len(correct))
        
    label_pred = []        
    for sentence in predictions:                     
        label_pred.append([idx2Label[element] for element in sentence])            

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
        
    prec = compute_precision_score(label_pred, label_correct)#compute_precision(label_pred, label_correct)
    rec = compute_precision_score(label_correct, label_pred)#compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def compute_precision_score(predicted, correct):
    
    prec = 0
    
    L = len(predicted)
    for idx in range(L):
        prec = prec + precision_score(predicted[idx], correct[idx], average='macro')    

    return prec/L
    
class Metrics(Callback):
    
    def __init__(self, train_data, idx2Label, pos_tag_index, dep_index, ner_index):
        self.valid_data = train_data    
        self.idx2Label = idx2Label
        self.pos_tag_index = pos_tag_index
        self.dep_index = dep_index
        self.ner_index = ner_index
        
        
    def on_train_begin(self, logs={}):         
        return
    
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):        
        return
    
    def on_epoch_end(self, epoch, logs={}):
        dataset = self.valid_data                
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, pos_tag, dep, ner, label = data
            input, output = transform([[tokens, casing, char, pos_tag, dep, ner, label]], max(2,len(label)), self.pos_tag_index, self.dep_index, self.ner_index)
            pred = self.model.predict(input, verbose=False)
#            pred = np.add(np.squeeze(pred[0]), np.flip(np.squeeze(pred[1]), axis=0))
            pred = pred.argmax(axis=-1)  # Predict the classes
            
            output = np.squeeze(output)
            output = np.argmax(output, axis=1)              
            correctLabels.append(output)
            
            pred = np.squeeze(pred)            
            predLabels.append(pred)                            
        
#        print('predLables', predLabels)
#        print('correctLables', correctLabels)
#        print(len(predLabels[1]))
#        print(len(correctLabels[1]))
        pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
        print("Dev-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_dev, rec_dev, f1_dev))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return