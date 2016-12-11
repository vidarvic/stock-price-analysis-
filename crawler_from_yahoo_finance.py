'''
Dec 5, 2016
Crawler from YAHOO finance
CPMAI
'''

from yahoo_finance import Share

yahoo = Share('YHOO')

# datatemp2 is for testing the Lasso model.
datatemp = yahoo.get_historical('2015-01-01', '2015-12-31')
datatemp2 = yahoo.get_historical('2016-01-01', '2016-01-31')

# Reverse the order of time.
data = []
for i in range(len(datatemp)):
	data.append(datatemp[len(datatemp)-i-1])
	
data_testSVR = []
for i in range(len(datatemp2)):
    data_testSVR.append(datatemp2[len(datatemp2)-i-1])

headers = ['Symbol','Date','High','Low','Close','Open','Volume','Adj_Close']

# Output the csv file.
import csv
with open('yahoo_2015.csv','w') as f:
	f_csv = csv.DictWriter(f, headers)
	f_csv.writeheader()
	f_csv.writerows(data)

# Output the json file.
import json
with open('yahoo_2015.json', 'w') as f:
	json.dump(data, f)

'''
Store feature in numpy-array form.
'''

# Data feature is T
# yahoo_target is for training model.
# yahoo_testSVR_target is for prediction.

import numpy as np

yahoo_target = np.empty((0,1))
for i in range(len(data)-1):
	yahoo_target = np.append(yahoo_target, float(data[i+1]['Close']))
	#yahoo_target = np.vstack((yahoo_target, np.array(data[i+1]['Close'])))
	
yahoo_testSVR_target = np.empty((0,1))
for i in range(len(data_testSVR)-1):
    yahoo_testSVR_target = np.append(yahoo_testSVR_target, data_testSVR[i+1]['Close'])

# Delete the non-number data(Symbol, Date).
# Data feature is T-1.
yahoo_feature_t1 = np.empty((0,6))
for i in range(len(data)-1):
	del data[i]['Symbol']
	del data[i]['Date']
	yahoo_feature_t1 = np.vstack((yahoo_feature_t1, np.array(data[i].values())))

yahoo_testSVR_feature_t1 = np.empty((0,6))
for i in range(len(data_testSVR)-1):
    del data_testSVR[i]['Symbol']
    del data_testSVR[i]['Date']
    yahoo_testSVR_feature_t1 = np.vstack((yahoo_testSVR_feature_t1, np.array(data_testSVR[i].values())))

# Convert the type of data.
yahoo_target = yahoo_target.astype(float)
yahoo_feature_t1 = yahoo_feature_t1.astype(float)
yahoo_testSVR_target = yahoo_testSVR_target.astype(float)
yahoo_testSVR_feature_t1 = yahoo_testSVR_feature_t1.astype(float)	

'''
Linear Regression
'''

#%matplotlib inline for ipython
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
y = yahoo_target
predicted = cross_val_predict(lr, yahoo_feature_t1, y, cv=10)

# Creat LR-pkl.
from sklearn.externals import joblib
joblib.dump(lr,"./lr_machine.pkl")
lr = joblib.load("./lr_machine.pkl")

# Predict the sample 3.
lr.fit(yahoo_feature_t1, y)
predict_y=lr.predict(yahoo_feature_t1[2])
predict_test=lr.predict(yahoo_testSVR_feature_t1)

print('score =',lr.score(yahoo_feature_t1, y))
print('score =',lr.score(yahoo_testSVR_feature_t1, yahoo_testSVR_target))

# Draw a picture.
plt.scatter(predicted, y, s=2)
plt.plot(predict_y, predict_y, 'ro')
plt.plot(predict_test, yahoo_testSVR_target, 'ro')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Measured')
plt.show()

'''
Machine Learning(Lasso)
'''

from sklearn.linear_model import Lasso

X = yahoo_feature_t1
clf2 = Lasso(alpha=0.1)
clf2.fit(X, y)

predict_y=clf2.predict(yahoo_feature_t1[2])
predict_test=clf2.predict(yahoo_testSVR_feature_t1)

predict=clf2.predict(X)
print 'score =',clf2.score(yahoo_feature_t1, y)
print('score =',clf2.score(yahoo_testSVR_feature_t1, yahoo_testSVR_target))

plt.scatter(predict,y,s=2)
plt.plot(predict_y, y[2], 'ro')
plt.plot(predict_test, yahoo_testSVR_target, 'ro')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Measured')