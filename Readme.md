# Adaptive Practice Recommender — Knowledge Tracing

An end-to-end machine learning system that predicts a student's probability of answering a question correctly on a given topic, and recommends which topics to practice next — ranked from weakest to strongest.

## Live Demo

- **Frontend (Streamlit)**: https://adaptive-practice-recommender-2vlca9ytfpgsww5ubefrm9.streamlit.app
- **Backend API (FastAPI on AWS EC2)**: http://3.135.1.9:8000/docs

## Problem

Given a student's history of practice attempts across topics, predict how likely they are to answer correctly on each topic next, and surface the topics they are weakest in — so they know exactly what to study next.

## Architecture


The frontend and backend are deployed independently: the frontend is a lightweight UI hosted on Streamlit Cloud, and the backend (model + business logic) runs in a Docker container on an AWS EC2 instance, so the heavier ML dependencies stay off the UI host.

## Dataset

RIIID / EdNet-style dataset (Kaggle "Riiid Answer Correctness Prediction") — real student-question interaction logs. A 1-million-row sample was used for this project (the full dataset has 100M+ rows); the pipeline is designed so it can be re-run on the full dataset with chunked/Dask processing.

## Feature Engineering

Three features were engineered from raw interaction logs, using only information available *before* each attempt (to avoid point-in-time leakage):

| Feature | Description |
|---|---|
| `topic_accuracy` | Rolling accuracy on this topic so far, based only on the student's past attempts |
| `time_since_last_attempt` | Time elapsed since the student last practiced this topic (captures memory decay / recency) |
| `ques_avg_accuracy` | Average difficulty of the question, computed as a global accuracy average across all students |

**Known limitation**: `ques_avg_accuracy` is computed as a global average across the *entire* dataset (train + test), rather than only from training data. This is a simplification that introduces mild data leakage. It was an intentional trade-off for this project's scope; a production version would compute this feature using only training-set statistics.

## Models Compared

| Model | Accuracy |
|---|---|
| Baseline (majority class) | 65.10% |
| Logistic Regression (unscaled features) | 65.10% |
| Logistic Regression (scaled features) | 71.01% |
| Random Forest | 68.90% |
| **XGBoost (best)** | **71.75%** |

XGBoost was selected as the final model and saved with `joblib`.

## Tech Stack

- **ML**: Python, Pandas, Scikit-learn, XGBoost
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deployment**: Docker, AWS EC2, Streamlit Community Cloud
- **Version control**: Git, GitHub

## Project Structure


## Running Locally

```bash
# Backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (in a separate terminal)
streamlit run app.py
```

## Running with Docker

```bash
docker build -t adaptive-recommender .
docker run -p 8000:8000 adaptive-recommender
```

## Deployment Notes

- The backend is containerized with Docker and deployed on an AWS EC2 instance (Ubuntu, t2.micro/t3.micro, free tier).
- `xgboost-cpu` (instead of `xgboost`) is used in production to avoid pulling in unnecessary GPU/CUDA dependencies, keeping the Docker image significantly smaller.
- The frontend is deployed separately on Streamlit Community Cloud and calls the AWS-hosted backend over HTTP.

## Future Improvements

- Scale the pipeline to the full 100M+ row dataset using chunked processing or Dask
- Compute `ques_avg_accuracy` using training-set-only statistics to remove the remaining leakage
- Add authentication and HTTPS to the backend API
- Add model monitoring / retraining pipeline