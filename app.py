import streamlit as st
import pandas as pd
import requests


summary= pd.read_csv("student_topic_summary.csv")

st.title("Adaptive Practice Recommender")

student_list= summary['user_id'].unique()
selected_student= st.selectbox("Select a student:", student_list)

if st.button("Get Recommendation"):
    url = f"http://127.0.0.1:8000/recommend/{selected_student}"
    response = requests.get(url)
    data = response.json()
    
    result_df = pd.DataFrame(data["recommendations"])
    st.write("Study these topics in order (weakest first):")
    st.dataframe(result_df)
    