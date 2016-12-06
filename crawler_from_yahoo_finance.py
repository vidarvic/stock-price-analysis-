'''
Dec 5, 2016
Crawler from YAHOO finance
Chi-Lun Mai
'''

from yahoo_finance import Share

yahoo = Share('YHOO')
data = yahoo.get_historical('2015-01-01', '2015-12-31')

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

# Store data in numpy-array form.
# Delete the non-number data(Volume, Symbol, Date).
import numpy as np
yahoo_2015 = np.empty((0,5))
for i in range(200):
    del data[i]['Volume']
    del data[i]['Symbol']
    del data[i]['Date']
	yahoo_2015 = np.vstack((yahoo_2015, np.array(data[i].values())))

print yahoo_2015