
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

prices = pd.read_excel('Commodity_Hedging_Model.xlsx',sheet_name='raw_data',index_col='date')
changes = prices - prices.shift(1)
changes = changes.iloc[1:,:]

unadj_hedge = prices['wti_spot'] - prices['wti_fut1'].shift(1)
unadj_hedge = unadj_hedge[1:]

changes['unadj_hedge'] = unadj_hedge
changes = changes.loc['2004-07-30':,]

dates = changes.index
windowsize = 24
nwindows = len(dates) - windowsize + 1

betavec = np.ndarray(nwindows)
for i in range(1,nwindows):
    date_0 = dates[i-1]
    date_n = dates[i-1 + windowsize-1]
    results = sm.ols(formula="jet_fuel ~ unadj_hedge", data=changes.loc[date_0:date_n,]).fit()
    betavec[i] = results.params[1]

unadj_hedge = changes.loc[dates[windowsize]:,'unadj_hedge']
hedge = unadj_hedge * betavec[1:]
jet_fuel = changes.loc[dates[windowsize]:,'jet_fuel']

err_vec = -(jet_fuel - hedge)

sd = err_vec.loc['2009-07-31':].std()

print(sd)