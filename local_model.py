import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Load data
iris = datasets.load_iris() 
X, y = datasets.load_iris( return_X_y = True)
  
# Splitting arrays or matrices into random train and test subsets

# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
                     'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
                     'species': iris.target})
#Model Creation
# clf = RandomForestClassifier(n_estimators = 100)  
clf = Pipeline([
             ('vect', CountVectorizer()),
             ('tfidf', TfidfTransformer()),
             ('clf', RandomForestClassifier()
        )])
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset


with open('model.joblib', 'wb') as f:
    joblib.dump(clf, f)


with open('model.joblib', 'rb') as f:
    predictor = joblib.load(f)

print("Testing following input: ")
clf.predict([[3, 3, 2, 2]])