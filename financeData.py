import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import holidays

"""
stockNum = '005930'
startDate = '2024-06-01'
endDate = '2024-07-02'
df = fdr.DataReader(stockNum,startDate,endDate)
"""
# df = fdr.DataReader('DJI') # 다우존스 지수 (DJI - Dow Jones Industrial Average)
df = fdr.DataReader('IXIC') # 나스닥 종합지수 (IXIC - NASDAQ Composite)
# df = fdr.DataReader('S&P500') # S&P500 지수 (NYSE)
print(df)