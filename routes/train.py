import json
import os
from typing import Optional

import joblib
import pandas as pd

from fastapi import HTTPException
from fastapi.responses import JSONResponse

import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter, Depends, File, HTTPException, Query
from schemas import PredictRequest, TrainResponse
from utils import generate_future_features, get_csv_data, preprocess_and_feature_engineer, sequential_predict, train_models_and_select

router = APIRouter()

# ------------------------------
# Globals (kept in memory + persisted)
# ------------------------------
DATA_DF: Optional[pd.DataFrame] = get_csv_data()
LABEL_ENCODERS = {}
FEATURES = []
TARGET = "quantity_sold"
MODEL_PATH = "../best_demand_model.pkl"
MODEL_META_PATH = "../model_meta.joblib"
MODEL_TYPE = None  # 'xgb' or 'lgb'
BEST_MODEL = None


# ------------------------------
# API Endpoints
# ------------------------------
@router.post("/start-training", response_model=TrainResponse, summary="Train models on uploaded CSV")
def train_pipeline():
    """
    Runs full preprocessing, feature engineering and trains XGB and LGB, selects best model.
    Returns metrics and a sample plot (history vs test predictions) serialized as Plotly JSON.
    """
    global DATA_DF, LABEL_ENCODERS, FEATURES, BEST_MODEL, MODEL_TYPE


    print('starting training pipeline...')
    if DATA_DF is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload-csv first.")

    df_proc = preprocess_and_feature_engineer(DATA_DF)
    print(df_proc.head())
    FEATURES = df_proc.columns.tolist()
    print(FEATURES)
    best_model, model_type, metrics = train_models_and_select(df_proc, FEATURES, TARGET)
    print(f"Best model: {model_type} with metrics: {metrics}")

    # persist
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump({"model_type": model_type, "features": FEATURES, "label_encoders": LABEL_ENCODERS}, MODEL_META_PATH)

    BEST_MODEL = best_model
    MODEL_TYPE = model_type

    # create a sample plot: last N actual vs predicted using test set
    df_proc = df_proc.sort_values('date').reset_index(drop=True)
    split_date = df_proc['date'].quantile(0.8)
    test_df = df_proc[df_proc['date'] >= split_date]
    X_test = test_df[FEATURES]
    X_test = X_test.drop(columns=['date'], errors='ignore')
    y_test = test_df[TARGET]
    print(X_test.info())
    preds = best_model.predict(X_test)

    fig = go.Figure()
    # plot actual mean per day (aggregate to reduce clutter)
    agg_actual = test_df.groupby('date')[TARGET].sum().reset_index()
    agg_pred = pd.DataFrame({'date': test_df['date'], 'pred': preds}).groupby('date')['pred'].sum().reset_index()
    fig.add_trace(go.Scatter(x=agg_actual['date'], y=agg_actual[TARGET], name='actual (sum/day)'))
    fig.add_trace(go.Scatter(x=agg_pred['date'], y=agg_pred['pred'], name='predicted (sum/day)'))
    fig.update_layout(title="Test Actual vs Predicted (daily aggregated)")
    fig_json = fig.to_json()
    fig_dict = json.loads(fig_json)

    return {"message": "Training complete", "model_type": model_type, "metrics": metrics, "sample_fig_json": fig_dict}


@router.post("/predict", summary="Predict future quantity for a product")
def predict(req: PredictRequest):
    """
    Predict future num_days for the given product (product_category, product, city).
    Returns a JSON with predictions and a Plotly figure JSON (history + future).
    """
    global DATA_DF, BEST_MODEL, MODEL_TYPE, LABEL_ENCODERS, FEATURES

    if DATA_DF is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload-csv first.")
    if BEST_MODEL is None:
        # attempt to load
        if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH):
            BEST_MODEL = joblib.load(MODEL_PATH)
            meta = joblib.load(MODEL_META_PATH)
            MODEL_TYPE = meta.get('model_type')
            FEATURES = meta.get('features')
            LABEL_ENCODERS = meta.get('label_encoders', {})
        else:
            raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    # prepare future features
    future_df, hist_df = generate_future_features(
        req.product_category, req.product, req.city, num_days=req.num_days,
        base_df=preprocess_and_feature_engineer(DATA_DF),
        price=req.price, discount=req.discount
    )

    # sequential predict
    future_preds = sequential_predict(BEST_MODEL, MODEL_TYPE, future_df, hist_df, FEATURES)

    # build a plotly graph: history (last 90 days) + future
    hist_plot = hist_df.sort_values('date').tail(90).copy()
    # use sum per date (if multiple products/cities) - but hist_df is for selected product so keep as-is
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_plot['date'], y=hist_plot['quantity_sold'], name='historical_quantity_sold'))
    fig.add_trace(go.Scatter(x=future_preds['date'], y=future_preds['predicted_quantity_sold'], name='predicted_future'))
    fig.update_layout(title=f"History & {req.num_days}-day Forecast for {req.product} in {req.city}")

    # feature importance (if available)
    fi = None
    try:
        if MODEL_TYPE == 'xgb':
            importance = BEST_MODEL.get_booster().get_score(importance_type='weight')
            fi = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
        else:
            if hasattr(BEST_MODEL, 'feature_importances_'):
                fi = pd.DataFrame({'feature': FEATURES, 'importance': BEST_MODEL.feature_importances_}).sort_values('importance', ascending=False)
        if fi is not None and not fi.empty:
            fig2 = px.bar(fi.head(20), x='feature', y='importance', title='Top 20 Feature Importances')
            fi_json = fig2.to_json()
        else:
            fi_json = None
    except Exception:
        fi_json = None

    # return predictions table and figures as JSON
    preds_table = future_preds[['date', 'predicted_quantity_sold']].to_dict(orient='records')
    return JSONResponse({
        "product": req.product,
        "city": req.city,
        "num_days": req.num_days,
        "predictions": preds_table,
        "figure_json": fig.to_json(),
        "feature_importance_json": fi_json
    })


@router.get("/model/status")
def model_status():
    global BEST_MODEL, MODEL_TYPE
    return {"model_loaded": BEST_MODEL is not None, "model_type": MODEL_TYPE}