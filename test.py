import pandas as pd

# 예시 데이터 프레임 생성
data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
    'Stock_Ratio': [0.5, -0.3, 0.0, 1.2]
}
df = pd.DataFrame(data)

print(df)
# 다음 행의 Stock_Ratio 값을 기준으로 label 열 추가
def determine_label(stock_ratio):
    if stock_ratio > 0:
        return 1
    elif stock_ratio < 0:
        return -1
    else:
        return 0

# shift() 메서드를 사용하여 다음 행의 Stock_Ratio 값을 가져옴
df['label'] = df['Stock_Ratio'].shift(-1).apply(determine_label)

print(df)