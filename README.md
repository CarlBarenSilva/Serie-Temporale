# Serie-Temporale
#Serie Temporale In Phython
#Revenue Forecasting
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Utility libraries
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error

# ARIMA libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('Regressione Cliente Bil v.csv')
df.head() #df
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)
df.dtypes
df = df.sort_index()
df['Billing'] = pd.to_numeric(df['Billing'], errors = 'coerce')
f = df[df['Cliente'] == "X"]
df.head()
df = df.drop(columns=['Cliente'])
df.head()
# set date column as index

df.index = pd.to_datetime(df['Mese'], format="%d/%m/%Y")
del df['Mese']
df.head()
df.columns
df= df.groupby('Mese').sum()
df
# Cost chart

sns.set()
plt.ylabel('Billing')
plt.xlabel('Mese')
plt.xticks(rotation = 45)

plt.plot(df.index,df['Billing'])
# Splitting Data into training & Test

train = df[df.index < pd.to_datetime("2022-12-01",format='%Y-%m-%d')]
test = df[df.index >= pd.to_datetime("2023-01-01",format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Billing')
plt.xlabel('Mese')
plt.xticks(rotation=45)
plt.title("Train/test Split")
plt.show()
y = train['Billing']

# Auto regressive models
ARMAmodel = SARIMAX(y,order = (36,0,3)) #very imp
ARMAmodel = ARMAmodel.fit()
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(train,color = "blue")
plt.plot(test, color = "red")
plt.plot(y_pred_out, color='green', label = 'ARMA Predictions')
plt.legend()
y
y_pred_df
arma_rmse = np.sqrt(mean_squared_error(test['Billing'].values, y_pred_df["Predictions"].values))
print("RMSE: ", arma_rmse)
test.head()
train
#2nd Model
from statsmodels.tsa.arima.model import ARIMA
ARIMAmodel = ARIMA(y, order = (36,0,5)) #factors for corelation, number of times raw observations are different and shock factor,moving average window
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.plot(test, color = "red")
plt.plot(train,color = "blue")
plt.xticks(rotation=45)
plt.legend()
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["Billing"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)c
y_pred_df
#3rd Model - Seasonal SARIMAX
SARIMAXmodel = SARIMAX(y, order = (1,1,5), seasonal_order=(1,1,0,2))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.plot(test, color = "red")
plt.plot(train,color = "black")
plt.xticks(rotation=45)
plt.legend()
y_pred_df
sarima_rmse = np.sqrt(mean_squared_error(test["Billing"].values, y_pred_df["Predictions"]))
print("RMSE: ",sarima_rmse)
test_month = pd.read_csv('test_data.csv')
test_month.index = pd.to_datetime(test_month['Data'], format="%d/%m/%Y")
del test_month['Data']
test_month
#Arima
pred= ARIMAmodel.predict(start = test_month.index[0], end = test_month.index[-1])

# Sarima
#pred= SARIMAXmodel.predict(start = test_month.index[0], end = test_month.index[-1])

#ARMA#pred = ARMAmodel.predict(start = test_month.index[0], end = test_month.index[-1])
test_month['Predictions'] = pred
test_month['Cliente'] = "X"
test_month
#change conc Name
test_month.to_csv(r"C:\Users\carbaren\Desktop\Esercizio\RegressioneConc_test.csv", index=False)
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

# Carica i dati
data = pd.Series([35.511, 47.949, 83.102, 61.774, 81.116, 117.117, 50.049, 70.711, 52.764, 39.017, 43.733])

# Plot ACF
plot_acf(data, lags=5)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Calcola il numero massimo di ritardi consentiti
max_lags = int(np.floor(data.shape[0] / 2))

# Plot PACF con lags <= max_lags
plot_pacf(data, lags=max_lags)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Carica i dati
data = df

# Esegui il test di Dickey-Fuller
result = adfuller(data)

# Stampare i risultati
print('Test Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
