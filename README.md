import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
Load data
df = pd.read_csv('stock_data.csv')
Handle missing values
df.fillna(df.mean(), inplace=True)
Remove duplicates
df.drop_duplicates(inplace=True)
Detect and treat outliers
Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Close'] < (Q1 - 1.5 * IQR)) | (df['Close'] > (Q3 + 1.5 * IQR)))]
Convert data types
df['Date'] = pd.to_datetime(df['Date'])
Encode categorical variables
df = pd.get_dummies(df, columns=['Category'])
Normalize features
scaler = MinMaxScaler()
df[['Close']] = scaler.fit_transform(df[['Close']])
Standardize features
scaler = StandardScaler()
df[['Volume']] = scaler.fit_transform(df[['Volume']])
