from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import numpy as np
import pylab as plt
import pandas as pd
from scipy.optimize import curve_fit

data = np.loadtxt('data8_6.txt').flatten()


f = lambda t, k, m, n : k * t ** 2 + m * t + n  
x0 = np.arange(100)
p, pcov = curve_fit(f, x0, data)
yh = f(x0, *p)
data_diff = data - yh
data1 = np.log(data)


data = pd.Series(data)
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1946Q1', '1970Q4'))

data_diff = pd.Series(data_diff)
data_diff.index = pd.Index(sm.tsa.datetools.dates_from_range('1946Q1', '1970Q4'))
model = auto_arima(data_diff, start_p=1, start_q=1, max_p=8, max_q=8, m=1,
                   start_P=0, seasonal=False, d=3, trace=True, information_criterion='aic',
                   error_action='ignore', 
                   suppress_warnings=True,
                   stepwise=False)
forecast = model.predict(8)
forecast = f(np.arange(100, 108), *p) + forecast

data_sum = data._append(forecast)
# plt.plot(data_sum)
# plt.plot(data)
# plt.savefig('figure8_6.png', dpi=500)
plt.show()
