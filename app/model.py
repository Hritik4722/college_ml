import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

feasibility_model = joblib.load(str(BASE_DIR / "models" / "feasibility_model2.pkl"))
cost_model = joblib.load(str(BASE_DIR / "models" / "cost_model.pkl"))
time_model = joblib.load(str(BASE_DIR / "models" / "time_model.pkl"))

FEATURE_NAMES = [
    'Estimated_Cost_USD',
    'Time_Estimate_Days',
    'Resource_Allocation_Score',
    'Risk_Assessment_Score',
    'Environmental_Impact_Score',
    'Historical_Cost_Deviation_%',
    'Stakeholder_Priority_Score',
    'Scope_Complexity_Numeric',
    'Project_Type_Building',
    'Project_Type_Power Plant',
    'Project_Type_Road',
    'Project_Type_Water Infra'
]

COST_FEATURE_NAMES = [
    'Scope_Complexity_Numeric',
    'Resource_Allocation_Score',
    'Risk_Assessment_Score',
    'Environmental_Impact_Score',
    'Historical_Cost_Deviation_%',
    'Stakeholder_Priority_Score',
    'Project_Type_Building',
    'Project_Type_Power Plant',
    'Project_Type_Road',
    'Project_Type_Water Infra'
]

TIME_FEATURE_NAMES = [
    'Resource_Allocation_Score',
    'Risk_Assessment_Score',
    'Environmental_Impact_Score',
    'Historical_Cost_Deviation_%',
    'Stakeholder_Priority_Score',
    'Scope_Complexity_Numeric',
    'Project_Type_Building',
    'Project_Type_Power Plant',
    'Project_Type_Road',
    'Project_Type_Water Infra'
]

def predict_feasibility(input_data):
    df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    prediction = feasibility_model.predict(df)[0]
    return prediction

def get_prediction_confidence(input_data):
    """Get the confidence score for the prediction."""
    try:
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        probas = feasibility_model.predict_proba(df)[0]
        return float(max(probas))
    except AttributeError:
        return 0.85

def predict_cost(input_data):
    """Predict estimated cost (USD) using the cost model."""
    df = pd.DataFrame([input_data], columns=COST_FEATURE_NAMES)
    prediction = cost_model.predict(df)[0]
    return float(prediction)

def predict_time(input_data):
    """Predict estimated time (days) using the time model."""
    df = pd.DataFrame([input_data], columns=TIME_FEATURE_NAMES)
    prediction = time_model.predict(df)[0]
    return float(prediction)
