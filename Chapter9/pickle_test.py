import pickle
import os
import numpy as np
from data.vectorizer import vect

clf = pickle.load(open(os.path.join('../data/pkl_objects', 'classifier.pkl'), 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]],
                                               np.max(clf.predict_proba(X))*100))
