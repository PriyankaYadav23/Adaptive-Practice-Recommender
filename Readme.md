# 📚 Adaptive Practice Recommender: Knowledge Tracing for Students

A machine learning system that predicts if a student will answer the next question correctly, then recommends which topics to study next, weakest first. Trained on real student practice data from the RIIID / EdNet dataset.

## 🧠 Machine Learning Pipeline

### Architecture Overview

The system works in four stages:

1. **Data Cleaning** - Load 1 million student interactions and keep only question attempts (lectures removed)
2. **Feature Engineering** - Build 3 features from each student's practice history
3. **Model Training** - Compare 4 models and pick the best one (XGBoost)
4. **Recommendation** - Predict the chance of a correct answer for each topic, then sort topics from weakest to strongest

### Technical Implementation

- **No Future Peeking**: Features use only the student's attempts *before* the current question, so the model never sees the answer it is predicting
- **Feature Scaling**: Features are scaled with `StandardScaler`, which fixed a model that was only guessing the majority class
- **Latest Snapshot**: A small summary table stores each student's latest state per topic, so the API stays fast

## 🔧 Feature Engineering

| Feature | What it means |
|---|---|
| `topic_accuracy` | How often the student got this topic right so far |
| `time_since_last_attempt` | How long since the student last practiced this topic (memory fades over time) |
| `ques_avg_accuracy` | How hard the question is, based on all students' answers |

**Known limitation**: `ques_avg_accuracy` is averaged over the whole dataset, so it uses a little future information. This is a common shortcut for question difficulty, and it is noted on purpose.

## 📊 Models Compared

| Model | Accuracy |
|---|---|
| Baseline (always guess "correct") | 65.10% |
| Logistic Regression (unscaled) | 65.10% |
| Logistic Regression (scaled) | 71.01% |
| Random Forest | 68.90% |
| **XGBoost (final model)** | **71.75%** |

## 🎬 Demo

**Live app:** https://adaptive-practice-recommender-2vlca9ytfpgsww5ubefrm9.streamlit.app

![Recommender Demo](examples/recommender.png)

*Pick a student from the dropdown, click "Get Recommendation", and see their topics ranked from weakest to strongest.*

> The backend runs on AWS EC2 only when needed (to keep the cost at $0), so the live app may not return results when the server is stopped.

**Example output for student 115:**

| Topic (part) | Chance of a correct answer |
|---|---|
| 3 | 16% ← study this first |
| 4 | 46% |
| 2 | 61% |
| 5 | 75% |
| 1 | 99% |

## 🛠️ Technology Stack

**Machine Learning**
- Python, Pandas
- scikit-learn
- XGBoost
- joblib

**Web Application & API**
- FastAPI
- Uvicorn
- Streamlit
- Requests

**Deployment**
- Docker
- AWS EC2
- Streamlit Community Cloud

## 🎮 Features

- **Weakest Topic First**: Topics are ranked by the predicted chance of a correct answer
- **Student History Aware**: Uses each student's own accuracy and practice timing
- **REST API**: `GET /recommend/{user_id}` returns the ranked topics as JSON
- **Separate Frontend and Backend**: The UI calls the API over HTTP, so each part can be deployed on its own
- **Zero-Cost Deployment**: EC2 is started only for demos, with a zero-spend budget alert as a safety net

## 📁 Project Structure

```
├── explore_data.ipynb          # Data cleaning, features, model training
├── model_xgb.pkl               # Trained XGBoost model
├── student_topic_summary.csv   # Latest features per student per topic
├── main.py                     # FastAPI backend (/recommend/{user_id})
├── app.py                      # Streamlit frontend
├── Dockerfile                  # Container for the backend
└── requirements.txt            # Python dependencies
```

The raw Kaggle files (`train.csv`, `questions.csv`) are not in the repo because of their size. Download them from the [Kaggle Riiid Answer Correctness Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction) page.

## 🏆 Technical Achievements

- **Real Data at Scale**: Worked with a 1-million-row sample of a 100M+ row real dataset
- **Leakage-Safe Features**: Rolling features built only from past attempts
- **Evidence-Based Model Choice**: 4 models tested on the same held-out data before picking XGBoost
- **Smaller Docker Image**: Switched to `xgboost-cpu` to drop a 342 MB unused GPU package, which fixed a "no space left" error on EC2
- **Full Product**: Notebook → API → web app → Docker → cloud, all working end to end

## 🔮 Future Improvements

- Run the pipeline on the full 100M+ rows with chunked processing or Dask
- Compute question difficulty from training data only, to remove the small leakage
- Show a friendly message when the backend is offline
- Add HTTPS and authentication to the API