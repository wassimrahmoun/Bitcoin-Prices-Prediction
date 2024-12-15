import streamlit as st , plotly.express as px
import time


if 'dates' not in st.session_state :
    st.error('Please chose correct dates in the homepage.')
else :
    dates = st.session_state.dates
    df = st.session_state.df

def visualize_data(df) :
  tab1 , tab2 = st.tabs(['Simple charts' , 'Advanced charts'])

  with tab1 :
    tab1_1 , tab1_2 = st.tabs(['Line chart' , 'Bar chart'])
    with tab1_1 :
     st.line_chart( data=df , y=['Open' , 'Close' , 'Low' , 'High'] , use_container_width=True )
     with tab1_2 :
       st.bar_chart( data=df , y=['Open' , 'Close' , 'Low' , 'High'] , use_container_width=True )


  with tab2 :
   tab2_1 , tab2_2 , tab2_3 = st.tabs(['Line chart' , 'Bar chart' , 'Scatter plot'])
   with tab2_1 :
      fig = px.line(df , y=['Open' , 'Close' , 'Low' , 'High'] , title='Bitcoin Prices over time' )
      st.plotly_chart(fig)
   with tab2_2 :
     fig = px.bar(df , y=['Open' , 'Close' , 'Low' , 'High'] , title='Bitcoin Prices over time' , opacity=0.4)
     st.plotly_chart(fig)
   with tab2_3 :
     fig = px.scatter(df , x='Volume' , y='Close' , title='Bitcoin Price vs Trading Volume' ) 
     st.plotly_chart(fig)

text = st.text("Drink a cup of water while you're waiting 💧")
progress_bar = st.progress(0 , text ="Drink a cup of water while you're waiting 💧") 
for i in range(100) :
    time.sleep(0.02)
    progress_bar.progress( i+1)

text.empty()
progress_bar.empty()
visualize_data(df)