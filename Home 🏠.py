import streamlit as st
import yfinance as yf
from datetime import datetime , date
# st.set_page_config(page_title=None , page_icon=None  , initial_sidebar_state='auto' , layout='wide')  # must be first line
menu_items = {
    'Get Help ' : 'https://elearn.esi-sba.dz/?redirect=0' ,
    'About' : '# Welcom to the SEDS Course'
    
}
st.set_page_config(page_title='Bitcoin' , page_icon='💵' , initial_sidebar_state='auto' , menu_items=menu_items , layout='wide')
# st.write('Hello World')     # 1_  prefix => page 1 ...etc ( specifies order )

# Fetching the Bitcoin price data
dates = None

if 'dates' in  st.session_state :  dates = st.session_state["dates"]

start_date = dates[0] if dates is not None  else datetime(2014,1,1)
end_date = dates[1] if dates is not None  else datetime(2018,1,1)


dates = st.slider('From when to when do you want to analyse the Bitcoin Prices ? ' , min_value=datetime(2014,1,1) , max_value=datetime(date.today().year , month=date.today().month , day=date.today().day) , 
                  value=(start_date , end_date))

if (dates[0] == dates[1] ) : 
    st.error('Make sure the Start Date is different from the Final One !' , icon='❌')

with st.sidebar :  # Making a loading spinner in the sidebar
   try :                          
    with st.spinner('Wait for it...') :
     df = yf.download("BTC-USD", start=dates[0] , end=dates[1])
   except Exception as e :
      st.error('Please Check Your Internet Connection ❌')

df.columns = df.columns.droplevel(1)


st.session_state["dates"] = dates
st.session_state["df"] = df


from sklearn.preprocessing import MinMaxScaler , LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from datetime import datetime

def model_predict(df, input):
    # Create a copy to avoid modifying the original dataframe
    df_model = df.reset_index()
    
    # Preprocess the date features
    df_model['Year'] = pd.to_datetime(df_model['Date']).dt.year
    df_model['Month'] = pd.to_datetime(df_model['Date']).dt.month
    df_model['Day'] = pd.to_datetime(df_model['Date']).dt.day
    
    # Drop the original Date column
    df_model = df_model.drop('Date', axis=1)
    
    # Remove any rows with NaN values
    df_model = df_model.dropna()

    # Separate features and target
    y = df_model['Close']
    x = df_model.drop(columns=['Close'])

    # Scale the features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=x.columns)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train the model
    model = SGDRegressor(learning_rate='constant', max_iter=8000, eta0=0.01, random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    r2_score = metrics.r2_score(y_true=y_test, y_pred=y_pred)
    print(f'R2 score is {r2_score}')

    # Preprocess the input data
    # Convert input to DataFrame
    input_df = pd.DataFrame(input)
    
    # Extract date features for input
    input_df['Year'] = pd.to_datetime(input_df['Date']).dt.year
    input_df['Month'] = pd.to_datetime(input_df['Date']).dt.month
    input_df['Day'] = pd.to_datetime(input_df['Date']).dt.day
    
    # Drop the original Date column from input
    input_df = input_df.drop('Date', axis=1)

    # Ensure all columns from training are present
    for col in x_train.columns:
        if col not in input_df.columns:
            raise ValueError(f"Missing column {col} in input data")

    # Select and order columns to match training data
    input_df = input_df[x_train.columns]

    # Scale the input data using the same scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction

# Example usage
# Assuming 'df' is your original dataframe with historical data
input = [{'Date' : datetime(2024,1,1) , 'Adj Close' : 91066.0078125 ,  'High' : 91066.0078125 , 'Low' : 91066.0078125 , 'Open' : 91066.0078125 , 'Volume' : 27889181179   }]
print(model_predict(df, input=input))