import streamlit as st
import yfinance as yf
from datetime import datetime , date 
from streamlit_option_menu import option_menu


if 'dates' not in st.session_state :
    st.error('Please chose correct dates in the homepage.')
else :
    dates = st.session_state.dates
    df = st.session_state.df

st.markdown("## Bitcoin prices 🪙")
st.markdown(f"### Between :green[{str(dates[0])[ : 10 ]}] to :green[{str(dates[1])[ : 10]}] ")    

st.dataframe(df , use_container_width=True)

# Statistics 

col1 , col2 , col3  = st.columns(3 , gap='small')
with col1 :
   global_lowest = round( min( df['Close'] ) , 1)
   st.metric("All time low " , f'${global_lowest}')
   

with col2 :
   st.metric("Average " , f'${ round( df['Close'].mean() , 1 )}')

  
with col3 :
   global_highest = round( max ( df['Close'] ) , 1 )
   croissance = round( ( (global_highest - global_lowest) / global_highest ) * 100 , 1 )
   print(croissance)
   st.metric("All time high " , f'${global_highest }' , f'{croissance}%')
