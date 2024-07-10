import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr

"""
##    날짜만 있는 CSV 생성
startDate = datetime(2000, 1, 1)
endDate = datetime(2024, 7, 9)
dateRange = pd.date_range(start=startDate, end=endDate)

data = {
  'Date' : dateRange,
  'Day' : [date.weekday() for date in dateRange]
}

df = pd.DataFrame(data)

df.to_csv('gather.csv', index=False)

##    다우존스 지수 추가
df = fdr.DataReader('DJI')
dict = df['Close'].to_dict()

df = pd.read_csv('gather.csv')
df['DJI_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)

##    나스닥 종합 지수 추가
df = fdr.DataReader('IXIC')
dict = df['Close'].to_dict()

df = pd.read_csv('gather.csv')
df['IXIC_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)

##    S&P500 지수 추가
df = fdr.DataReader('S&P500')
dict = df['Close'].to_dict()

df = pd.read_csv('gather.csv')
df['S&P500_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)

##    WTI 선물 추가
df = fdr.DataReader('CL=F')
dict = df['Close'].to_dict()
df = pd.read_csv('gather.csv')
df['CL_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    브렌트유 선물 추가
df = fdr.DataReader('BZ=F')
dict = df['Close'].to_dict()
df = pd.read_csv('gather.csv')
df['BZ_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    천연가스 선물 추가
df = fdr.DataReader('NG=F')
dict = df['Close'].to_dict()
df = pd.read_csv('gather.csv')
df['NG_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    금 선물 추가
df = fdr.DataReader('GC=F')
dict = df['Close'].to_dict()
df = pd.read_csv('gather.csv')
df['GC_Close'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    코스피 추가
df = pd.read_excel('./지표/코스피 및 코스닥 지수.xlsx')
df.columns = ['Date','KOSPI','KOSDAQ']
df = df[6:]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
dict = df['KOSPI'].to_dict()
df = pd.read_csv('gather.csv')
df['KOSPI'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    코스닥 추가
df = pd.read_excel('./지표/코스피 및 코스닥 지수.xlsx')
df.columns = ['Date','KOSPI','KOSDAQ']
df = df[6:]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
dict = df['KOSDAQ'].to_dict()
df = pd.read_csv('gather.csv')
df['KOSDAQ'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    주요 장기 시장금리
colList = ['A','B','C','D']
nameList = ['LTMIR_A','LTMIR_B','LTMIR_C','LTMIR_D']
for i in range(0,4):
  df = pd.read_excel('./지표/주요 장기 시장금리.xlsx')
  df.columns = ['Date','A','B','C','D']
  df = df[6:]
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  dict = df[colList[i]].to_dict()
  df = pd.read_csv('gather.csv')
  df[nameList[i]] = df['Date'].map(dict)
  df.to_csv('gather.csv', index=False)
  print(df)

##    주요 단기 시장금리
colList = ['A','B','C','D']
nameList = ['STMIR_A','STMIR_B','STMIR_C','STMIR_D']
for i in range(0,4):
  df = pd.read_excel('./지표/주요 단기 시장금리.xlsx')
  df.columns = ['Date','A','B','C','D']
  df = df[6:]
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  dict = df[colList[i]].to_dict()
  df = pd.read_csv('gather.csv')
  df[nameList[i]] = df['Date'].map(dict)
  df.to_csv('gather.csv', index=False)
  print(df)

##    원/유로 환율
df = pd.read_excel('./지표/원_유로 및 원_위안 환율.xlsx')
df.columns = ['Date','Euro','Wian']
df = df[6:]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
dict = df['Euro'].to_dict()
df = pd.read_csv('gather.csv')
df['Euro'] = df['Date'].map(dict)
df.to_csv('gather.csv', index=False)
print(df)

##    원/달러, 엔 환율
colList = ['Dollar','Yen']
nameList = ['Dollar','Yen']
for i in range(0,4):
  df = pd.read_excel('./지표/원_달러 및 원_엔 환율.xlsx')
  df.columns = ['Date','Dollar','Yen']
  df = df[6:]
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  dict = df[colList[i]].to_dict()
  df = pd.read_csv('gather.csv')
  df[nameList[i]] = df['Date'].map(dict)
  df.to_csv('gather.csv', index=False)
  print(df)

##    주가 데이터 추가
# df1 = pd.read_csv('./지표/삼성전자주가1.csv')
# df1['일자'] = pd.to_datetime(df1['일자'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
# df2 = pd.read_csv('./지표/삼성전자주가2.csv',encoding='cp949')
# df1 = df1.append(df2, ignore_index=True)
# df1.to_csv('삼성전자주가.csv', encoding='utf-8')
# print(df1)
# df.columns = ['Date','Stock_Close','Stock_Sum','Stock_Ratio','Stock_']
nameList = ['Stock_Close','Stock_Sum','Stock_Ratio','Stock_Start','Stock_High','Stock_Low','Stock_Vol','Stock_Money','Stock_Market','NotUse']
for i in range(1,10):
  df = pd.read_csv('./지표/삼성전자주가.csv')
  df = df[0:]
  df['일자'] = pd.to_datetime(df['일자'])
  df.set_index('일자',inplace=True)
  dict = df[df.columns[i]].to_dict()
  df = pd.read_csv('gather.csv')
  df[nameList[i-1]] = df['Date'].map(dict)
  df.to_csv('gather.csv', index=False)
  print(df)
"""
df = pd.read_csv('gather.csv')
del df['NotUse']
df.to_csv('gather.csv')