'''
Dec 5, 2016
Crawler from YAHOO finance
CPMAI
'''

from yahoo_finance import Share

yahoo = Share('YHOO')
datatemp = yahoo.get_historical('2015-01-01', '2015-12-31')
data = []

# Reverse the order of time.
for i in range(len(datatemp)):
	data.append(datatemp[len(datatemp)-i-1])

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
import numpy as np
yahoo_target = np.empty((0,1))
for i in range(len(data)-1):
	yahoo_target = np.append(yahoo_target, float(data[i+1]['Close']))
	#yahoo_target = np.vstack((yahoo_target, np.array(data[i+1]['Close'])))

# Delete the non-number data(Symbol, Date).
# Data feature is T-1.
yahoo_feature_t1 = np.empty((0,6))
for i in range(len(data)-1):
	del data[i]['Symbol']
	del data[i]['Date']
	yahoo_feature_t1 = np.vstack((yahoo_feature_t1, np.array(data[i].values())))

# Convert the type of data.
yahoo_target = yahoo_target.astype(float)
yahoo_feature_t1 = yahoo_feature_t1.astype(float)	
'''
Machine Learning
'''

#%matplotlib inline
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
y = yahoo_target
predicted = cross_val_predict(lr, yahoo_feature_t1, y, cv=10)

from sklearn.externals import joblib
joblib.dump(lr,"./lr_machine.pkl")


lr=joblib.load("./lr_machine.pkl")

lr.fit(yahoo_feature_t1, y)
predict_y=lr.predict(yahoo_feature_t1[2])


plt.scatter(predicted, y, s=2)
plt.plot(predict_y, predict_y, 'ro')

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

plt.xlabel('Predicted')
plt.ylabel('Measured')