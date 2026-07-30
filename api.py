"""
🎓 Student Exam Pass/Fail Predictor — FastAPI Backend
------------------------------------------------------
Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import os

# ── Load models ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    student_df = pd.read_csv(os.path.join(MODELS_DIR, "student_data.csv"))
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False
    _load_error = str(e)

FEATURES = [
    "study_hours_per_day", "attendance_rate", "previous_gpa", "sleep_hours",
    "assignments_done_pct", "parent_education", "tutoring",
    "extracurricular", "internet_access", "gender_encoded",
]

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎓 Student Pass/Fail Predictor API",
    description=(
        "Predict whether a student will **pass** or **fail** their exam "
        "using Logistic Regression or Random Forest. "
        "Run the notebook first to generate the model files."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    study_hours_per_day:   float = Field(..., ge=0,   le=10,  example=4.5,  description="Average study hours per day")
    attendance_rate:       float = Field(..., ge=40,  le=100, example=78.0, description="Class attendance percentage")
    previous_gpa:          float = Field(..., ge=0.0, le=4.0, example=3.1,  description="Previous semester GPA")
    sleep_hours:           float = Field(..., ge=3,   le=10,  example=7.0,  description="Average nightly sleep hours")
    assignments_done_pct:  float = Field(..., ge=0,   le=100, example=85.0, description="Percentage of assignments submitted")
    parent_education:      int   = Field(..., ge=0,   le=2,   example=1,    description="0=High School, 1=Some College, 2=University Degree")
    tutoring:              int   = Field(..., ge=0,   le=1,   example=1,    description="Receives tutoring: 0=No, 1=Yes")
    extracurricular:       int   = Field(..., ge=0,   le=1,   example=0,    description="Extracurricular activities: 0=No, 1=Yes")
    internet_access:       int   = Field(..., ge=0,   le=1,   example=1,    description="Home internet access: 0=No, 1=Yes")
    gender_encoded:        int   = Field(..., ge=0,   le=1,   example=1,    description="Gender: 0=Female, 1=Male")

    @field_validator("parent_education", "tutoring", "extracurricular", "internet_access", "gender_encoded")
    @classmethod
    def must_be_binary_or_trinary(cls, v, info):
        field = info.field_name
        if field == "parent_education" and v not in (0, 1, 2):
            raise ValueError("parent_education must be 0, 1, or 2")
        if field != "parent_education" and v not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "study_hours_per_day": 5.0,
                "attendance_rate": 82.0,
                "previous_gpa": 3.2,
                "sleep_hours": 7.5,
                "assignments_done_pct": 90.0,
                "parent_education": 2,
                "tutoring": 1,
                "extracurricular": 0,
                "internet_access": 1,
                "gender_encoded": 0,
            }
        }
    }


class PredictionResult(BaseModel):
    model:            str
    prediction:       int
    result:           str
    pass_probability: float
    fail_probability: float
    confidence:       str
    risk_level:       str


class BatchInput(BaseModel):
    students: list[StudentInput]
    model: Literal["logistic_regression", "random_forest", "both"] = "random_forest"


# ── Helpers ────────────────────────────────────────────────────────────────
def _to_df(student: StudentInput) -> pd.DataFrame:
    return pd.DataFrame([student.model_dump()])[FEATURES]


def _risk(prob: float) -> str:
    if prob >= 0.80: return "Low Risk"
    if prob >= 0.60: return "Moderate Risk"
    if prob >= 0.40: return "High Risk"
    return "Very High Risk"


def _confidence(prob: float) -> str:
    diff = abs(prob - 0.5) * 2      # 0 → random, 1 → certain
    if diff >= 0.70: return "High"
    if diff >= 0.35: return "Medium"
    return "Low"


def _predict(model, name: str, X: pd.DataFrame) -> PredictionResult:
    pred   = int(model.predict(X)[0])
    probs  = model.predict_proba(X)[0]
    p_pass = round(float(probs[1]), 4)
    p_fail = round(float(probs[0]), 4)
    return PredictionResult(
        model=name,
        prediction=pred,
        result="PASS" if pred == 1 else "FAIL",
        pass_probability=p_pass,
        fail_probability=p_fail,
        confidence=_confidence(p_pass),
        risk_level=_risk(p_pass),
    )


def _check_models():
    if not MODELS_OK:
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded. Run the notebook first. Error: {_load_error if not MODELS_OK else ''}"
        )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["General"])
def root():
    """Interactive landing page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Student Pass/Fail Predictor API</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0c29;
         background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
         min-height: 100vh; color: #fff; padding: 2rem; }
  .card { background: rgba(255,255,255,0.07); border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.12); padding: 2rem;
          max-width: 700px; margin: 2rem auto; backdrop-filter: blur(10px); }
  h1 { font-size: 2rem; margin-bottom: 0.4rem; }
  .sub { color: #a0aec0; margin-bottom: 1.5rem; }
  .badge { display:inline-block; background:#4f46e5; color:#fff;
           padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem;
           margin-right:0.4rem; margin-bottom:0.4rem; }
  .endpoint { background: rgba(0,0,0,0.3); border-radius: 10px;
              padding: 1rem 1.2rem; margin-bottom: 0.8rem;
              border-left: 3px solid #818cf8; }
  .method { font-weight: 700; color: #34d399; margin-right: 0.5rem; font-size: 0.85rem; }
  .path { font-family: monospace; color: #f0f0f0; }
  .desc { color: #94a3b8; font-size: 0.88rem; margin-top: 0.3rem; }
  .btn { display:inline-block; margin-top:1.5rem; background:#4f46e5;
         color:#fff; padding:0.7rem 1.5rem; border-radius:8px;
         text-decoration:none; font-weight:600; transition:0.2s; }
  .btn:hover { background:#4338ca; }
  .btn-outline { background:transparent; border:1px solid #818cf8; margin-left:0.8rem; color:#818cf8; }
  .status { display:flex; align-items:center; gap:0.5rem; margin-bottom:1.2rem; }
  .dot { width:10px;height:10px;border-radius:50%;background:#34d399;
         animation:pulse 2s infinite; }
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
</style>
</head>
<body>
<div class="card">
  <h1>🎓 Student Predictor API</h1>
  <p class="sub">Predict exam pass/fail using ML — FastAPI backend</p>
  <div class="status">
    <div class="dot"></div>
    <span style="color:#34d399;font-size:0.9rem">API Online</span>
  </div>
  <span class="badge">Logistic Regression</span>
  <span class="badge">Random Forest</span>
  <span class="badge">FastAPI</span>
  <span class="badge">Scikit-learn</span>

  <div style="margin-top:1.5rem">
    <div class="endpoint">
      <span class="method">GET</span><span class="path">/health</span>
      <div class="desc">Check API and model status</div>
    </div>
    <div class="endpoint">
      <span class="method">POST</span><span class="path">/predict/random-forest</span>
      <div class="desc">Predict with Random Forest (recommended)</div>
    </div>
    <div class="endpoint">
      <span class="method">POST</span><span class="path">/predict/logistic-regression</span>
      <div class="desc">Predict with Logistic Regression</div>
    </div>
    <div class="endpoint">
      <span class="method">POST</span><span class="path">/predict/compare</span>
      <div class="desc">Run both models and compare side-by-side</div>
    </div>
    <div class="endpoint">
      <span class="method">POST</span><span class="path">/predict/batch</span>
      <div class="desc">Predict for multiple students at once</div>
    </div>
    <div class="endpoint">
      <span class="method">GET</span><span class="path">/dataset/stats</span>
      <div class="desc">Summary statistics of the training dataset</div>
    </div>
    <div class="endpoint">
      <span class="method">GET</span><span class="path">/model/features</span>
      <div class="desc">List of required features and their descriptions</div>
    </div>
  </div>

  <a class="btn" href="/docs">📖 Swagger UI</a>
  <a class="btn btn-outline" href="/redoc">📄 ReDoc</a>
</div>
</body>
</html>
"""


@app.get("/health", tags=["General"])
def health():
    """API and model health check."""
    return {
        "status": "ok",
        "models_loaded": MODELS_OK,
        "available_models": ["logistic_regression", "random_forest"] if MODELS_OK else [],
        "dataset_rows": len(student_df) if MODELS_OK else 0,
    }


@app.get("/model/features", tags=["Model Info"])
def model_features():
    """Describe each input feature."""
    return {
        "features": [
            {"name": "study_hours_per_day",  "type": "float", "range": "0–10",   "description": "Average study hours per day"},
            {"name": "attendance_rate",      "type": "float", "range": "40–100", "description": "Class attendance percentage"},
            {"name": "previous_gpa",         "type": "float", "range": "0–4.0",  "description": "Previous semester GPA"},
            {"name": "sleep_hours",          "type": "float", "range": "3–10",   "description": "Average nightly sleep hours"},
            {"name": "assignments_done_pct", "type": "float", "range": "0–100",  "description": "Percentage of assignments submitted"},
            {"name": "parent_education",     "type": "int",   "range": "0|1|2",  "description": "0=High School, 1=Some College, 2=University Degree"},
            {"name": "tutoring",             "type": "int",   "range": "0|1",    "description": "Receives tutoring support"},
            {"name": "extracurricular",      "type": "int",   "range": "0|1",    "description": "Participates in extracurricular activities"},
            {"name": "internet_access",      "type": "int",   "range": "0|1",    "description": "Has home internet access"},
            {"name": "gender_encoded",       "type": "int",   "range": "0|1",    "description": "0=Female, 1=Male"},
        ]
    }


@app.get("/dataset/stats", tags=["Data"])
def dataset_stats():
    """Summary statistics of the training dataset."""
    _check_models()
    desc = student_df.describe().round(3).to_dict()
    return {
        "total_rows":     len(student_df),
        "pass_count":     int(student_df["passed"].sum()),
        "fail_count":     int((student_df["passed"] == 0).sum()),
        "pass_rate_pct":  round(student_df["passed"].mean() * 100, 2),
        "feature_stats":  desc,
    }


@app.post("/predict/random-forest", response_model=PredictionResult, tags=["Predict"])
def predict_rf(student: StudentInput):
    """Predict using the **Random Forest** classifier."""
    _check_models()
    return _predict(rf_model, "random_forest", _to_df(student))


@app.post("/predict/logistic-regression", response_model=PredictionResult, tags=["Predict"])
def predict_lr(student: StudentInput):
    """Predict using **Logistic Regression**."""
    _check_models()
    return _predict(lr_model, "logistic_regression", _to_df(student))


@app.post("/predict/compare", tags=["Predict"])
def predict_compare(student: StudentInput):
    """Run **both models** and return a side-by-side comparison."""
    _check_models()
    X = _to_df(student)
    lr_res = _predict(lr_model, "logistic_regression", X)
    rf_res = _predict(rf_model, "random_forest", X)
    agree  = lr_res.result == rf_res.result
    return {
        "logistic_regression": lr_res,
        "random_forest":       rf_res,
        "models_agree":        agree,
        "consensus":           lr_res.result if agree else "UNCERTAIN — models disagree",
        "avg_pass_probability": round((lr_res.pass_probability + rf_res.pass_probability) / 2, 4),
    }


@app.post("/predict/batch", tags=["Predict"])
def predict_batch(payload: BatchInput):
    """
    Predict for **multiple students** in a single request.
    `model` can be `"logistic_regression"`, `"random_forest"`, or `"both"`.
    """
    _check_models()
    if len(payload.students) > 100:
        raise HTTPException(status_code=400, detail="Max 100 students per batch request.")

    results = []
    for i, student in enumerate(payload.students):
        X = _to_df(student)
        if payload.model == "both":
            entry = {
                "student_index":       i,
                "logistic_regression": _predict(lr_model, "logistic_regression", X),
                "random_forest":       _predict(rf_model, "random_forest", X),
            }
        else:
            model = rf_model if payload.model == "random_forest" else lr_model
            entry = {
                "student_index": i,
                "prediction":    _predict(model, payload.model, X),
            }
        results.append(entry)

    pass_count = sum(
        1 for r in results
        for res in ([r.get("prediction")] if "prediction" in r
                    else [r["random_forest"]])
        if res and res.result == "PASS"
    )

    return {
        "total_students":    len(results),
        "model_used":        payload.model,
        "pass_count":        pass_count,
        "fail_count":        len(results) - pass_count,
        "pass_rate_pct":     round(pass_count / len(results) * 100, 1),
        "results":           results,
    }
