import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
model = joblib.load(str(BASE_DIR / "models" / "feasibility_model2.pkl"))

# Feature names used during training
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

def predict_feasibility(input_data):
    # Convert to DataFrame with feature names to avoid sklearn warning
    df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    prediction = model.predict(df)[0]
    return prediction

def get_prediction_confidence(input_data):
    """Get the confidence score for the prediction."""
    try:
        # Convert to DataFrame with feature names to avoid sklearn warning
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        # Get probability estimates if available
        probas = model.predict_proba(df)[0]
        return float(max(probas))
    except AttributeError:
        # If model doesn't support predict_proba, return a default confidence
        return 0.85
