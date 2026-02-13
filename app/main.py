from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pathlib import Path
import time

from model import predict_feasibility, get_prediction_confidence, predict_cost, predict_time
from visualize import generate_all_visualizations
import joblib

app = FastAPI()

# Get the directory of the current file
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

# Ensure static directory exists
STATIC_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # No content - silently ignore favicon requests

model = joblib.load(str(MODELS_DIR / "feasibility_model2.pkl"))

# Project types for one-hot encoding (drop_first=True drops 'Bridge' alphabetically)
# When Project_Type is 'Bridge', all encoded values will be 0
PROJECT_TYPES_ENCODED = ['Building', 'Power Plant', 'Road', 'Water Infra']

FEATURES = [
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

@app.get("/")
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    Project_Type: str = Form(...),
    Estimated_Cost_USD: float = Form(...),
    Time_Estimate_Days: int = Form(...),
    Resource_Allocation_Score: float = Form(...),
    Risk_Assessment_Score: float = Form(...),
    Environmental_Impact_Score: float = Form(...),
    Historical_Cost_Deviation_: float = Form(...),
    Stakeholder_Priority_Score: float = Form(...),
    Scope_Complexity_Numeric: int = Form(...)
):
    # One-hot encode the Project_Type (Bridge is the reference/dropped category)
    project_type_encoded = [1 if pt == Project_Type else 0 for pt in PROJECT_TYPES_ENCODED]
    
    input_data = [
        Estimated_Cost_USD,
        Time_Estimate_Days,
        Resource_Allocation_Score,
        Risk_Assessment_Score,
        Environmental_Impact_Score,
        Historical_Cost_Deviation_,
        Stakeholder_Priority_Score,
        Scope_Complexity_Numeric
    ] + project_type_encoded

    # Store input data for form repopulation
    input_dict = {
        'Project_Type': Project_Type,
        'Estimated_Cost_USD': Estimated_Cost_USD,
        'Time_Estimate_Days': Time_Estimate_Days,
        'Resource_Allocation_Score': Resource_Allocation_Score,
        'Risk_Assessment_Score': Risk_Assessment_Score,
        'Environmental_Impact_Score': Environmental_Impact_Score,
        'Historical_Cost_Deviation_': Historical_Cost_Deviation_,
        'Stakeholder_Priority_Score': Stakeholder_Priority_Score,
        'Scope_Complexity_Numeric': Scope_Complexity_Numeric
    }

    result = predict_feasibility(input_data)
    confidence = get_prediction_confidence(input_data)

    label_map = {
        0: "Not Feasible",
        1: "Feasible",
        2: "Borderline"
    }

    result_label = label_map[result]
    
    # Generate all visualizations
    generate_all_visualizations(model, FEATURES, input_data, result_label)
    
    # Add timestamp to prevent caching
    cache_bust = int(time.time())

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result_label,
            "confidence": confidence,
            "input_data": input_dict,
            "feature_plot": f"/static/feature_importance.png?v={cache_bust}",
            "radar_plot": f"/static/radar_chart.png?v={cache_bust}",
            "gauge_plot": f"/static/gauge_chart.png?v={cache_bust}",
            "distribution_plot": f"/static/distribution_chart.png?v={cache_bust}",
            "active_tab": "feasibility"
        }
    )

@app.post("/predict-cost")
def predict_cost_route(
    request: Request,
    Project_Type: str = Form(...),
    Scope_Complexity_Numeric: int = Form(...),
    Resource_Allocation_Score: float = Form(...),
    Risk_Assessment_Score: float = Form(...),
    Environmental_Impact_Score: float = Form(...),
    Historical_Cost_Deviation_: float = Form(...),
    Stakeholder_Priority_Score: float = Form(...)
):
    # One-hot encode the Project_Type (Bridge = all zeros)
    project_type_encoded = [1 if pt == Project_Type else 0 for pt in PROJECT_TYPES_ENCODED]

    # Build input in EXACT training order (must match COST_TIME_FEATURE_NAMES)
    input_data = [
        Scope_Complexity_Numeric,
        Resource_Allocation_Score,
        Risk_Assessment_Score,
        Environmental_Impact_Score,
        Historical_Cost_Deviation_,
        Stakeholder_Priority_Score
    ] + project_type_encoded

    # Get prediction
    cost_result = predict_cost(input_data)

    # Store input data for form repopulation
    cost_input_dict = {
        'Project_Type': Project_Type,
        'Scope_Complexity_Numeric': Scope_Complexity_Numeric,
        'Resource_Allocation_Score': Resource_Allocation_Score,
        'Risk_Assessment_Score': Risk_Assessment_Score,
        'Environmental_Impact_Score': Environmental_Impact_Score,
        'Historical_Cost_Deviation_': Historical_Cost_Deviation_,
        'Stakeholder_Priority_Score': Stakeholder_Priority_Score
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "cost_result": f"{cost_result:,.2f}",
            "cost_raw": cost_result,
            "cost_input_data": cost_input_dict,
            "active_tab": "cost"
        }
    )


@app.post("/predict-time")
def predict_time_route(
    request: Request,
    Project_Type: str = Form(...),
    Scope_Complexity_Numeric: int = Form(...),
    Resource_Allocation_Score: float = Form(...),
    Risk_Assessment_Score: float = Form(...),
    Environmental_Impact_Score: float = Form(...),
    Historical_Cost_Deviation_: float = Form(...),
    Stakeholder_Priority_Score: float = Form(...)
):
    # One-hot encode the Project_Type (Bridge = all zeros)
    project_type_encoded = [1 if pt == Project_Type else 0 for pt in PROJECT_TYPES_ENCODED]

    # Build input in EXACT training order (must match TIME_FEATURE_NAMES)
    # Time model used df.drop() — Scope_Complexity_Numeric comes AFTER the scores
    input_data = [
        Resource_Allocation_Score,
        Risk_Assessment_Score,
        Environmental_Impact_Score,
        Historical_Cost_Deviation_,
        Stakeholder_Priority_Score,
        Scope_Complexity_Numeric
    ] + project_type_encoded

    # Get prediction
    time_result = predict_time(input_data)

    # Store input data for form repopulation
    time_input_dict = {
        'Project_Type': Project_Type,
        'Scope_Complexity_Numeric': Scope_Complexity_Numeric,
        'Resource_Allocation_Score': Resource_Allocation_Score,
        'Risk_Assessment_Score': Risk_Assessment_Score,
        'Environmental_Impact_Score': Environmental_Impact_Score,
        'Historical_Cost_Deviation_': Historical_Cost_Deviation_,
        'Stakeholder_Priority_Score': Stakeholder_Priority_Score
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "time_result": f"{time_result:,.0f}",
            "time_raw": time_result,
            "time_input_data": time_input_dict,
            "active_tab": "time"
        }
    )
