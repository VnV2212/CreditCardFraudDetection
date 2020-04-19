# Always divide into 3 sets when dealing with skew datasts i.e probaility of one event exceeds other by a large margin.
# training data can consist of 60% of normal data and cross validation data will consist of 20% of normal and 50% of
# anomalous data. same goes for test data.

# get mu and sigma on training data, epsilon on CV and test finally.
import sklearn
import random
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns


# returns probability density function.
def mulVarGauss(x, mean, sig):
    p = multivariate_normal(mean=mean, cov=sig)
    return p.pdf(x)


# returns mean and covariance matrix.
def getGauss(data):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data.T)
    return mu, sigma


ccset = pd.read_csv("creditcard.csv")
x = ccset.drop(['Class'], axis=1)
y = ccset['Class']

# using ExtraTree classifier to understand feature importance i.e which features highly affects our output.
model = ExtraTreesClassifier()
model.fit(x, y)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# dropping unnecessary features.
ccset.drop(
    ['Amount', 'Time', 'V19', 'V21', 'V1', 'V2', 'V6', 'V5', 'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20',
     'V15', 'V13', 'V8'],
    axis=1, inplace=True)

cnt0 = 0
cnt1 = 0

for i in ccset['Class']:
    if i == 1:
        cnt1 += 1
    else:
        cnt0 += 1

x1 = [cnt0, cnt1]
x2 = ['n-fraud', 'fraud']

train_strip_v1 = ccset[ccset['Class'] == 1]
train_strip_v0 = ccset[ccset['Class'] == 0]

# This can be obtained by using a chosen set of epsilon to test on cross validation set and choose the one which gives
# best results. Here we get this epsilon.
epsilon = 1.0527717316e-70

Normal_len = len(train_strip_v0)
Anomolous_len = len(train_strip_v1)

m = Anomolous_len // 2
mplus = m + 1

# dividing in training, testing and cv sets.
train_cv_v1 = train_strip_v1[: m]
train_test_v1 = train_strip_v1[mplus:Anomolous_len]

m = (Normal_len * 60) // 100
mplus = m + 1

cm = (Normal_len * 80) // 100
cmplus = cm + 1

train_fraud = train_strip_v0[:m]
train_cv = train_strip_v0[mplus:cm]
train_test = train_strip_v0[cmplus:Normal_len]

train_cv = pd.concat([train_cv, train_cv_v1], axis=0)
train_test = pd.concat([train_test, train_test_v1], axis=0)

train_cv_y = train_cv["Class"]
train_test_y = train_test["Class"]

train_cv.drop(labels=["Class"], axis=1, inplace=True)
train_fraud.drop(labels=["Class"], axis=1, inplace=True)
train_test.drop(labels=["Class"], axis=1, inplace=True)

# getting optimal epsilon and probability matrix.
mu, sigma = getGauss(train_fraud)
prob_train = mulVarGauss(train_fraud, mu, sigma)
prob_cv = mulVarGauss(train_cv, mu, sigma)
prob_test = mulVarGauss(train_test, mu, sigma)

# making predictions on test set.
predictions = []
test1 = 0
for i in prob_test:
    if i < epsilon:
        predictions.append(1)
        test1 += 1
    else:
        predictions.append(0)
predictions = np.asarray(predictions)
train_test_y = np.asarray(train_test_y)

# checking the performance with below metrics
print(" Recall is ", recall_score(train_test_y, predictions, average="binary"))
print(" Precision is ", precision_score(train_test_y, predictions, average="binary"))
print(" F1 score is ", f1_score(train_test_y, predictions, average="binary"))
# Accuracy score doesn't matter here as this is a skew dataset.
print(" Accuracy score is ", accuracy_score(train_test_y, predictions))

# classifier2
ccset = pd.read_csv('creditcard.csv')
y = ccset['Class']
x = ccset.drop(['Class', 'Amount', 'Time', 'V19', 'V21', 'V1', 'V2', 'V6', 'V5', 'V28', 'V27', 'V26', 'V25', 'V24', 'V23', 'V22', 'V20',
     'V15', 'V13', 'V8'], axis=1)
x = x.values
y = y.values
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2)
classifier2 = RandomForestClassifier(n_estimators=100)
classifier2.fit(x_train2, y_train2)

predictions2 = classifier2.predict(x_test2)

print(" Recall is ", recall_score(y_test2, predictions2))
print(" Precision is ", precision_score(y_test2, predictions2))
print(" F1 score is ", f1_score(y_test2, predictions2))
# Accuracy score doesn't matter here aws this is a skew dataset.
print(" Accuracy score is ", accuracy_score(y_test2, predictions2))
print(confusion_matrix(y_test2, predictions2))
print(classification_report(y_test2, predictions2))

classfier3
classifier3 = IsolationForest(n_estimators=100, max_samples=100)
classifier3.fit(x_train)

tmp = classifier3.predict(x_test)
predictions3 = []
for i in tmp:
    if i == 1:
        predictions3.append(0)
    else:
        predictions3.append(1)
predictions3 = np.asarray(predictions3)

print(" Recall is ", recall_score(y_test, predictions3))
print(" Precision is ", precision_score(y_test, predictions3))
print(" F1 score is ", f1_score(y_test, predictions3))
# Accuracy score doesn't matter here aws this is a skew dataset.
print(" Accuracy score is ", accuracy_score(y_test, predictions3))
print(confusion_matrix(y_test, predictions3))


#Classifier4
classifier4 = KNeighborsClassifier()
classifier4.fit(x_train2, y_train2)
predictions4 = classifier4.predict(x_test)
predictions4 = np.asarray(predictions4)

print(" Recall is ", recall_score(y_test, predictions4))
print(" Precision is ", precision_score(y_test, predictions4))
print(" F1 score is ", f1_score(y_test, predictions4))
# Accuracy score doesn't matter here aws this is a skew dataset.
print(" Accuracy score is ", accuracy_score(y_test, predictions4))
print(confusion_matrix(y_test, predictions4))


final_predictions = []

for i in range(len(x_test)):
    tmp = [predictions[i], predictions3[i], predictions4[i]]
    final_predictions.append(max(set(tmp), key = tmp.count))

final_predictions = np.asarray(final_predictions)

print(" Recall is ", recall_score(y_test, final_predictions))
print(" Precision is ", precision_score(y_test, final_predictions))
print(" F1 score is ", f1_score(y_test, final_predictions))
# Accuracy score doesn't matter here aws this is a skew dataset.
print(" Accuracy score is ", accuracy_score(y_test, final_predictions))
print(confusion_matrix(y_test, final_predictions))