import numpy as np
import pandas as pd

##Load CSV data
data = pd.read_csv("labelled_data.csv", names=['count','density','traffic'])
##print(data.head())

#define samples
X = []
y = []
for _,row in data.iterrows():
    X.append([row['count'],row['density']])
    y.append(int(row['traffic']))
##print(X[0:5])
##print(y[0:5])
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

##naive bayes classifier
from sklearn.naive_bayes import GaussianNB
print("=======================================")
print("Naive Bayes Classifier Accuracy")
print("---------------------------------------")
clf = GaussianNB()
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
print("Average Accuracy = %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


##knn classifier
from sklearn.neighbors import KNeighborsClassifier
print("=======================================")
print("KNN Classifier")
print("---------------------------------------")
for j in range(1,11):
    clf = KNeighborsClassifier(n_neighbors = j)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("k = %d, Average Accuracy : %0.2f (+/- %0.2f)" % (j, scores.mean(), scores.std() * 2))

##svm classifier
from sklearn import svm
print("=======================================")
print("SVM Classifier")
print("---------------------------------------")
accuracy = []
deviation = []
for j in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVC(kernel=j)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("kernel = %s, Average Accuracy : %0.2f (+/- %0.2f)" % (j, scores.mean(), scores.std() * 2))
    accuracy.append(scores.mean())
    
##decision tree
from sklearn import tree
print("=======================================")
print("Decision Tree Classifier")
print("---------------------------------------")
clf = tree.DecisionTreeClassifier()
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
print("Average Accuracy = %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

##radom forest
from sklearn import ensemble
print("=======================================")
print("Random Forest Classifier")
print("---------------------------------------")
for j in range(1,11):
    clf = ensemble.RandomForestClassifier(n_estimators = j)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("estimators = %d, Average Accuracy : %0.2f (+/- %0.2f)" % (j, scores.mean(), scores.std() * 2))
print("=======================================")
