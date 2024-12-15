import streamlit as st
from datetime import datetime , date
from sklearn.preprocessing import MinMaxScaler , LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import pandas as pd

st.set_page_config(layout='centered')

if 'df' not in st.session_state :
    st.error('Please make sure to select valid dates in the Home page.')

df = st.session_state.df

def model_predict(df, input):
    # Extracting Date fron index into a new Column 'Date'
    df_model = df.reset_index()
    
    # Preprocess the date features ( instead of using LabelEncoder)
    df_model['Year'] = pd.to_datetime(df_model['Date']).dt.year
    df_model['Month'] = pd.to_datetime(df_model['Date']).dt.month
    df_model['Day'] = pd.to_datetime(df_model['Date']).dt.day
    
    # Drop the original Date column
    df_model = df_model.drop('Date', axis=1)
    
    df_model = df_model.dropna()

    y = df_model['Close']
    x = df_model.drop(columns=['Close'])

    # Scale the features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled, columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Training the model
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
    
    input_df = input_df.drop('Date', axis=1)

    # Ensure all columns from training are present
    for col in x_train.columns:
        if col not in input_df.columns:
            raise ValueError(f"Missing column {col} in input data")

    # Select and order columns to match training data
    input_df = input_df[x_train.columns]

    # Scaling the input data with the same Scaler
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    
    return (prediction , r2_score)


Date = st.date_input("Select a date" , value=date.today() )
col1 , col2 , col3 , col4 , col5 = st.columns(5)

with col1 :
  adj_close = st.number_input('Adj_Close') 

with col2 :
  High = st.number_input('High') 

with col3 :
  Low = st.number_input('Low') 

with col4 :
  Open = st.number_input('Open') 

with col5 :
  Volume = st.number_input('Volume') 

if st.button('Predict' , type='primary')   : 
    try :
      if ( Low <= 0  or High <= 0 or Open <= 0 or adj_close <= 0 or Volume <= 0 ) :
        raise Exception('Error')
      
      with st.spinner('Wait for it...') :
       input = [{'Date' : pd.to_datetime(Date) , 'Adj Close' : adj_close , 'High' : High , 'Low' : Low , 'Open' : Open , 'Volume' : Volume   }]
       prediction , r2_score = model_predict(df=df , input=input)
       st.markdown(f'The predicted closing price is **:blue[{prediction[0]}]**')
       st.markdown(f'The R2_Score of the model is **:green[{r2_score}]**')
    except Exception as e :
        st.error('Please check your inputs ' , icon='❌')
       
