import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

stop = stopwords.words('english')
score = 'f1_macro'

def loaddata():
    data_path = input('Please input your data path: ')
    data_path = data_path + '\homework.json'
    return pd.read_json(data_path, lines=True)
    #return pd.read_json(r'C:\Users\yluo\Downloads\homework.json\homework.json', lines=True)
    
    
def prepData(data):
    data.fillna('0', inplace=True)
    data = data.replace({'fnord': '1', 'cat': '1', '-1': '1', '-2': '1', '1.0':'1', '0.0': '0'})
    
    return data
    
def prepNewRec(data):
    rec = ''
    for key, value in data.items():
        if isinstance(value, str):
            rec += ' ' + data[key]
    
    return [rec]
    
def predictEmotion(data, newRec):
    res = {}
    perf_score = {}
    
    df = prepData(data)
    text_hl_sum = df['headline'] + ' ' + df['summary']
    
    processedRec = prepNewRec(newRec)
    
    # create transformer 
    vectorizer = CountVectorizer()
    encoder = LabelEncoder()
    
    # tokenize and build vocabulary_
    vectorizer.fit(text_hl_sum)
    
    # encode document
    X = vectorizer.transform(text_hl_sum)
    #print('training data - transformed matrix shape: ', X.shape)
    
    vect_new = vectorizer.transform(processedRec)
    #print('new records - transformed matrix shape: ', vect_new.shape)
    #print()

    
    for i in range(10):
        emotion = 'emotion_'+str(i)
        
        y = df[emotion]
        #print (X.shape[0], len(y))
        
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
        # resample data set to increase minority calss
        #resample = SMOTE()
        resample = RandomOverSampler()
        X_train_new, y_train_new = resample.fit_sample(X_train, y_train)
        
        clf_pipe = Pipeline([('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB())])

        tuned_parameters = {
            'tfidf__norm': ('l1', 'l2'),
            'tfidf__use_idf': (False, True),
            'mnb__alpha': [1, 0.1, 0.01]
        }
        
        np.errstate(divide='ignore')
        clf = GridSearchCV(clf_pipe, tuned_parameters, cv=10, scoring=score)
        clf.fit(X_train_new, y_train_new)
        
        perf_score[emotion] = clf.best_score_
        
        print()
        print('~~~~~~~~~~~~~~~~~ %s ~~~~~~~~~~~~~~~' % emotion)
        print()
        
        print('Best score: %0.4f with parameters %r' % (clf.best_score_, clf.best_params_))
        print()
        
        print('Detailed model performance score with parameters to correctly predict the results:')
        for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                                     clf.cv_results_['std_test_score'], 
                                     clf.cv_results_['params']):
            print('%0.4f +/-%0.04f with parameters %r' % (mean, std * 2, params))
        print()
        
        
        print("Detailed classification report (scores were computed on evaluation data set):")
        print()
        print(classification_report(y_test, clf.predict(X_test), digits=4))
        print()
        
        
        ####### predict the emotion for new headline and summary 
        pred = clf.predict(vect_new)
        #print (emotion, pred)
        res[emotion] = int(pred[0])
        
    return res

'''    
if __name__ == '__main__':
    rec = { "headline": "Irma Live Updates: Now a Tropical Storm, System Heads North", "summary": "After days of frantic preparation, Floridians awoke Monday to destruction that fell short of the direst forecasts, but still faced life-altering damage.", "worker_id": 1 }
    #emotion_res = prepNewRec(rec)
    emotion_res = predictEmotion(loaddata(), rec)
    print ('new predicted: ', emotion_res)
    print ()
'''    
    