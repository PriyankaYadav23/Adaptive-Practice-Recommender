import os

import streamlit as st
import pandas as pd
import requests

# Backend address in ONE place. After an EC2 restart, update the default IP here,
# or set an API_URL secret on Streamlit Cloud instead of editing code.
API_URL = os.environ.get("API_URL", "http://3.135.1.9:8000")

summary = pd.read_csv("student_topic_summary.csv")

st.title("Adaptive Practice Recommender")

student_list = summary['user_id'].unique()
selected_student = st.selectbox("Select a student:", student_list)

if st.button("Get Recommendation"):
    url = f"{API_URL}/recommend/{selected_student}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        st.error("The recommendation server is offline right now "
                 "(it runs on demand to keep cloud cost at $0). Please try again later.")
        st.stop()

    data = response.json()
    result_df = pd.DataFrame(data["recommendations"])
    st.write("Study these topics in order (weakest first):")
    st.dataframe(result_df)
