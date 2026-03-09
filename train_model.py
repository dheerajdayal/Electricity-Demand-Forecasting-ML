import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Electircity_consumbtionp_prediction\dataset\household_power_consumption.csv",na_values=['?', 'nan', ''],
                 low_memory=False,
                 header=0, 
                 nrows=10000)

df['Sub_metering_3']=df['Sub_metering_3'].fillna(df['Sub_metering_3'].mean())

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['IsWeekend'] = df['DayOfWeek'] >= 5

le=LabelEncoder()

cols=[ 'DayOfWeek', 'Month', 'IsWeekend']
for c in cols:
  df[c]=le.fit_transform(df[c])
df['Rolling_3'] = df['Global_active_power'].rolling(window=3).mean()
df['Rolling_5'] = df['Global_active_power'].rolling(window=5).mean()

 
rolling_cols = ['Rolling_3', 'Rolling_5']
df[rolling_cols] = df[rolling_cols].fillna(df[rolling_cols].mean())
print(df.head(20))
if 'index' in df.columns:
    df = df.drop(columns=['index'])
df = df.reset_index(drop=True)


feature=[
'Global_reactive_power', 'Voltage',
'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Hour',
'DayOfWeek', 'Month', 'IsWeekend','Rolling_3', 'Rolling_5']

Timefeature=['Date','Time','Datetime']

x=df[feature]
y=df['Global_active_power']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

modelpipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("model",LinearRegression())
])

modelpipeline.fit(x_train,y_train)

y_pred = modelpipeline.predict(x_test)

print(r2_score(y_test,y_pred))


y_train_pred = modelpipeline.predict(x_train)


y_test_pred = modelpipeline.predict(x_test)


train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train R²:", train_r2)
print("Test R²:", test_r2)

pickle.dump(modelpipeline,open('Electricity consumption prediction.ppkl','wb'))
