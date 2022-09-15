import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
#ML model
from keras.models import load_model
import streamlit as st
start='2010-01-01'
end='2019-12-31'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)
#describe data
st.subheader('Data from 2010-1019')
st.write(df.describe())

#visualization

st.subheader('closing price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs Time chart with 100ma')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs Time chart with 100ma & 200ma')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Spliting Data into Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#load models
model=load_model('keras_model.h5')

#testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing , ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test=np.array(x_test),np.array(y_test)
#marking predictions
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Prediction vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted, 'r',label='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)