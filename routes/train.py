import json
import os
from typing import Optional

import joblib
import pandas as pd

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter, Depends, File, HTTPException, Query
from schemas import PredictRequest, TrainResponse
from utils import generate_future_features, get_csv_data, preprocess_and_feature_engineer, sequential_predict, train_models_and_select
import boto3
import os

router = APIRouter()

# ------------------------------
# Globals (kept in memory + persisted)
# ------------------------------
DATA_DF: Optional[pd.DataFrame] = get_csv_data()
LABEL_ENCODERS = {}
FEATURES = []
TARGET = "quantity_sold"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'latest_model.pkl'))
MODEL_META_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_TYPE = 'xgb' # 'xgb' or 'lgb'
# BEST_MODEL = joblib.load(MODEL_PATH)
BEST_MODEL = None

# Helper to get latest CSV name from S3 uploads folder
def get_latest_csv_key():
    try:
        from settings import settings
        s3 = boto3.client(
            "s3",
            region_name=getattr(settings, "AWS_REGION", None),
            aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
        )
        bucket_name = getattr(settings, "S3_BUCKET_NAME", None)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='uploads/')
        if 'Contents' not in response or not response['Contents']:
            return None
        latest_obj = max(response['Contents'], key=lambda x: x['LastModified'])
        return latest_obj['Key']
    except Exception as e:
        print(f"Error getting latest CSV from S3: {e}")
        return None
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

    df_proc, LABEL_ENCODERS, FEATURES, TARGET = preprocess_and_feature_engineer(DATA_DF)
    print(df_proc.head())
    print(FEATURES)
    best_model, model_type, metrics = train_models_and_select(df_proc, FEATURES, TARGET)
    print(f"Best model: {model_type} with metrics: {metrics}")

    # Get CSV filename for model naming
    csv_key = get_latest_csv_key()
    if csv_key is None:
        raise HTTPException(status_code=400, detail="No CSV found in S3 uploads/")
    csv_filename = os.path.basename(csv_key)
    model_filename = f"{os.path.splitext(csv_filename)[0]}_model.pkl"
    local_model_path = os.path.join(MODEL_META_PATH, model_filename)
    meta_filename = f"{os.path.splitext(csv_filename)[0]}_meta.pkl"
    local_meta_path = os.path.join(MODEL_META_PATH, meta_filename)

    # persist locally
    joblib.dump(best_model, local_model_path)
    joblib.dump({"model_type": model_type, "features": FEATURES, "label_encoders": LABEL_ENCODERS, "processed_df": df_proc}, local_meta_path)

    # Upload model to S3
    try:
        from settings import settings
        s3 = boto3.client(
            "s3",
            region_name=getattr(settings, "AWS_REGION", None),
            aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
        )
        with open(local_model_path, "rb") as f:
            s3.upload_fileobj(f, settings.S3_BUCKET_NAME, f"models/{model_filename}")
        with open(local_meta_path, "rb") as f:
            s3.upload_fileobj(f, settings.S3_BUCKET_NAME, f"models/{meta_filename}")
    except Exception as e:
        print(f"Error uploading model/meta to S3: {e}")

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
    model_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    MODEL_META_PATH = os.path.join(model_folder_path, 'e88d8e9598f34806b67d8dbcee7232fc_meta.pkl')
    MODEL_PATH = os.path.join(model_folder_path, 'e88d8e9598f34806b67d8dbcee7232fc_model.pkl')
    BEST_MODEL = joblib.load(MODEL_PATH)

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
            df_proc = meta.get('processed_df')
            print(BEST_MODEL, MODEL_TYPE, FEATURES, LABEL_ENCODERS, df_proc)
        else:
            raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")

    # getting requirements
    base_df, LABEL_ENCODERS, FEATURES, target = preprocess_and_feature_engineer(DATA_DF)
    # prepare future features
    future_df, hist_df, FEATURES = generate_future_features(
        req.product_category, req.product, req.city, num_days=req.num_days,
        base_df=base_df,
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

    future_preds['date'] = future_preds['date'].astype(str)
    # return predictions table and figures as JSON
    preds_table = future_preds[['date', 'predicted_quantity_sold']].to_dict(orient='records')
    print(future_preds[['date', 'predicted_quantity_sold']])
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