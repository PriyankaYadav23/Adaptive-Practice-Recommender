from fastapi import FastAPI
import pandas as pd
import joblib 

app= FastAPI()
model=joblib.load("model_xgb.pkl")
summary=pd.read_csv("student_topic_summary.csv")

def recommend_topics(user_id):
    student_data= summary[summary['user_id']==user_id]
    features = student_data[['topic_accuracy', 'time_since_last_attempt', 'ques_avg_accuracy']]
    probabilities = model.predict_proba(features)[: , 1]
    student_data= student_data.copy()
    student_data['predicted_prob']= probabilities
    ranked= student_data.sort_values('predicted_prob')
    return ranked[['part','predicted_prob']].to_dict(orient='records')

@app.get("/recommend/{user_id}")
def get_recommend(user_id:int):
    result=recommend_topics(user_id)
    return {"user_id": user_id ,"recommendations":result}