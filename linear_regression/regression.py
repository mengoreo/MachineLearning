import pandas as pd
import quandl
quandl.ApiConfig.api_key = 'Y5J7DEBJiTkXpGLpCyju'
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'

# Replace the not a number data to -9999
# instead of getting rid of the column
df.fillna(-99999, inplace=True)

# Use data that came 10 days ago to predict today ??
forecast_out = int(math.ceil(0.3*len(df)))
print(forecast_out)
# Shifting backwards
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
# Normalize the value
X = preprocessing.scale(X)

# Predicting part
X_lately = X[-forecast_out:]

# Training part
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])



# 80% as the training datesets, and the rest as testing datesets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LinearRegression
#clf = LinearRegression()

# SVM
#clf = svm.SVR(kernel='poly')

# Training
#clf.fit(X_train, y_train)

# Pickling classifier
# with open('linear_regression_classifier.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# Scaling classifier
with open('linear_regression_classifier.pickle', 'rb') as f:
    clf = pickle.load(f)


# and testing
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Set the forecast column to NAN
df['Forecast'] = np.nan

# Create time stamp

## Last date in the datasets
last_date = df.iloc[-1].name

last_unix = last_date.timestamp()
one_day = 86400 # seconds
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Replace the next_date rows with NAN + forecast
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
