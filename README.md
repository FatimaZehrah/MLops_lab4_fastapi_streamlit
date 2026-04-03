# MLops_lab4_fastapi_streamlit

## IE7374 – MLOps Lab 4: FastAPI + Streamlit ML Inference Dashboard

This project builds on a **FastAPI-based ML inference service** and adds a **Streamlit dashboard** to interact with the deployed model in a clean, user-friendly way.

It demonstrates a practical MLOps pattern:
- **FastAPI backend** hosts a trained ML model and exposes REST endpoints
- **Streamlit frontend** consumes those endpoints to provide interactive inference + lightweight monitoring

---

## Project Overview

The goal of this lab is to operationalize a machine learning classifier as a service and provide an interactive interface to:
- check service health
- view model metadata (model card)
- submit inference requests using multiple input methods
- display prediction + confidence
- track latency and inference history (session-based)

---

## Problem Statement

Deploy an ML classifier as a web service that:
1. accepts structured input
2. returns predictions and confidence scores
3. exposes a health endpoint for monitoring
4. provides model metadata for transparency
5. includes a UI layer for user-friendly interaction

---

## Model Details
- **Dataset:** Iris dataset
- **Model Type:** Scikit-learn classifier
- **Prediction Output:** `prediction` (class label) + `confidence`
- **Serialization:** Pickle (`.pkl`)

---

## Key Features

### Backend (FastAPI)
- `GET /` → basic service message
- `GET /health` → service health monitoring
- `POST /predict` → returns prediction + confidence
- `GET /model-info` → returns model metadata (model card)

### Frontend (Streamlit Dashboard)
- **Dual input modes**
  - Manual sliders
  - JSON file upload
- **Latency tracking** per request + average latency metric
- **Prediction history** table stored per session (`st.session_state`)
- **Low-confidence warning** using a threshold slider
- **Model Info panel** (reads from `/model-info`)

---

## Custom Enhancements (New Factors)

To ensure this lab implementation is not identical to a base template, the following enhancements were added:

- Added a **Model Info endpoint** (`GET /model-info`) and a Streamlit **Model Card panel**
- Implemented **two input modes**: manual sliders + JSON upload
- Added **prediction history** using Streamlit `session_state`
- Added **latency measurement** per prediction + average latency summary
- Added **input validation + friendly error handling** in the dashboard

---

## Tools and Technologies Used
- **Python**
- **FastAPI**
- **Uvicorn**
- **Streamlit**
- **Scikit-learn**
- **Pydantic**
- **Requests**
- **Pandas**
- **Pytest** (backend tests, if present)
- **GitHub** / **GitHub Actions** (if workflows included)

---

## Project Structure

```bash
MLops_lab4_fastapi_streamlit/
│── backend/
│   ├── src/
│   ├── model/
│   ├── test/
│   └── requirements.txt
│
│── frontend/
│   ├── Dashboard.py
│   ├── requirements.txt
│   ├── sample_inputs/
│   └── assets/
│
│── README.md
│── .gitignore
```

## How to Run Locally

### Prerequisites
- Python 3.x installed
- Two terminals (one for backend, one for frontend)

---

### 1) Start the FastAPI Backend

Open **Terminal 1** in the repo root and run:

```bash
cd backend
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

Backend should run at:
- `http://127.0.0.1:8000`

Verify these endpoints:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/model-info`
- `http://127.0.0.1:8000/docs`

> If the model file is missing, regenerate it (optional):
```bash
python src/train.py
```
### 2) Start the Streamlit Dashboard (Frontend)


Open **Terminal 2** in the repo root and run:

```bash
cd frontend
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Dashboard.py
```
Streamlit should open at:

- http://localhost:8501

### 3) Run a Prediction in the UI

In the Streamlit dashboard:

- Confirm **Backend Status = UP ✅**
- Confirm **Model Info** is visible

Then use either input mode:

#### Option A: Manual (Sliders)
1. Select **Manual (Sliders)**
2. Adjust feature values
3. Click **Run Prediction**
4. Confirm you see:
   - prediction + confidence
   - latency (ms)
   - a new row in prediction history

#### Option B: Upload JSON
1. Select **Upload JSON**
2. Upload a JSON file like:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
3.Click Run Prediction

4.Confirm the same outputs appear (response + latency + history update)

### 4) Run Backend Tests (Optional)

If you have tests in `backend/test/`, run:

```bash
cd backend
# activate venv first if not active
pytest
```
## Author
Fatima Zehrah
Master’s in Data Analytics Engineering
Northeastern University
