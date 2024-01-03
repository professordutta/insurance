import streamlit as st
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
import pickle


model = pickle.load(open('model.pkl','rb'))


st.header('Insurance Prediction using Machine Learning', divider='rainbow')



# navigation bar
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select options below to know more:', ['Exploratory Data Analysis','Insurance Prediction'])


df = pd.read_csv('insurance.csv')
@st.cache_data(experimental_allow_widgets=True)
def stats():
    st.header(':blue[Exploratory Data Anaylsis]')
    st.dataframe(df)
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data='https://github.com/professordutta/insurance/blob/main/insurance.csv',
        file_name='data.csv',
        mime='text/csv',
    )
    #plot 1
    fig1 = px.histogram(df, x="sex", color="smoker", barmode='group')
    st.plotly_chart(fig1, use_container_width = True)
    
    #plot 2
    fig2 = px.histogram(df, x="children", color="smoker", barmode='group')
    st.plotly_chart(fig2, use_container_width = True)

    #plot 3
    fig3 = px.histogram(df, x="region", color="smoker", barmode='group')
    st.plotly_chart(fig3, use_container_width = True)

    #plot 4
    fig4 = px.box(df, x="smoker", y="charges", color="sex")
    st.plotly_chart(fig4, use_container_width = True)

    #plot 5
    hist_data = [df["age"]]
    group_labels = ['distplot']
    fig5 = ff.create_distplot(hist_data, group_labels)
    st.plotly_chart(fig5, use_container_width = True)

    #plot 6
    hist_data = [df["bmi"]]
    group_labels = ['distplot']
    fig6 = ff.create_distplot(hist_data, group_labels)
    st.plotly_chart(fig6, use_container_width = True)

    #plot 7
    fig7 = px.scatter(
    df, x='age', y='charges', opacity=0.65,color='smoker',
    trendline='ols', trendline_color_override='darkblue')
    st.plotly_chart(fig7, use_container_width = True)

    #plot 8
    fig8 = px.scatter(df, x="bmi", y="charges", color="smoker")
    st.plotly_chart(fig8, use_container_width = True)


    
def enterdata():

    def predict(age, sex, bmi, children, smoke, region):
        # female = 0, male = 1
        if sex == "Male":
            sex = 1
        else:
            sex = 0 

        # yes = 1, no = 0
        if smoke == 'Yes':
            smoke = 1
        else:
            smoke = 0
        
        # northeast = 0, southwest = 1, southeast = 2, northwest = 3
        if region == 'northeast':
            region = 0
        elif region == 'southwest':
            region = 1
        elif region == 'southeast':
            region = 2
        else:
            region = 3
        

        # calling model 
        input=np.array([[age, sex, bmi, children, smoke, region]]).astype(np.float64)
        pred = model.predict(input)
        return pred

    age = st.number_input("Enter your age", value=None, placeholder="Enter your age...")
    st.write('The current number is ', age)

    sex = st.selectbox(
   "Select your gender",
   ("Male", "Female"),
   index=None,
   placeholder="Select your gender...",)
    st.write('You selected:', sex)   

    bmi = st.number_input("Enter your bmi", value=None, placeholder="Enter your bmi...")
    st.write('The current number is ', bmi)

    children = st.selectbox(
   "Select your no. of children",
   ("0", "1",'2','3','4','5'),
   index=None,
   placeholder="Select your no. of children...",)
    st.write('You selected:', children)

    smoke = st.selectbox(
   "Do you smoke",
   ("Yes", "No"),
   index=None,
   placeholder="Do you smoke...",)
    st.write('You selected:', smoke)    

    region = st.selectbox(
   "Select your region",
   ("southeast", "southwest",'northwest','northeast'),
   index=None,
   placeholder="Select your region...",)
    st.write('You selected:', region)   

    if st.button("Predict"):
        output = predict(age, sex, bmi, children, smoke, region)
        st.write(output)
    

if options == 'Exploratory Data Analysis':
    stats()

elif options == 'Insurance Prediction':
    enterdata()
