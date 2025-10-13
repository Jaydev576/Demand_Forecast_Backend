# router.py  (modified: uploads to S3 on train complete, download from S3 for predict)
import json
import os
import tempfile
from typing import Optional
from urllib.parse import urlparse

import joblib
import pandas as pd

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter

from schemas import PredictRequest, TrainResponse
from utils import generate_future_features, get_csv_data, preprocess_and_feature_engineer, sequential_predict, train_models_and_select

import boto3
from datetime import datetime

# DB imports - adjust module path if different
from db import SessionLocal
import crud
from settings import settings

router = APIRouter()

# ------------------------------
# Globals (kept in memory + persisted)
# ------------------------------
DATA_DF: Optional[pd.DataFrame] = None
LABEL_ENCODERS = {}
FEATURES = []
TARGET = "quantity_sold"
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_TYPE = 'xgb' # 'xgb' or 'lgb'
BEST_MODEL = None
BEST_MODEL_META = None

# S3 client (uses credentials from settings / env)
_s3_client = boto3.client(
    "s3",
    region_name=getattr(settings, "AWS_REGION", None),
    aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
    aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
)

# ------------------------------
# Helpers
# ------------------------------
def parse_s3_uri(s3_uri: str):
    """
    Accepts 's3://bucket/key' or 'bucket/key' and returns (bucket, key)
    """
    if not s3_uri:
        return None, None
    if s3_uri.startswith("s3://"):
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    else:
        # allow passing "bucket/key"
        parts = s3_uri.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
    return bucket, key

def upload_file_to_s3(local_path: str, s3_key: str):
    """ Upload local file to configured S3 bucket under s3_key (uploads/ or models/). """
    bucket = getattr(settings, "S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3 bucket not configured in settings.S3_BUCKET_NAME")
    _s3_client.upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"

def download_s3_to_tempfile(s3_uri: str):
    """ Download S3 object to a temporary file and return the temp file path. Caller should remove file. """
    bucket, key = parse_s3_uri(s3_uri)
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    try:
        _s3_client.download_file(bucket, key, tmp.name)
    except Exception as e:
        # cleanup on failure
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        raise
    return tmp.name

# Helper to get latest CSV name from S3 uploads folder
def get_latest_csv_key():
    try:
        bucket_name = getattr(settings, "S3_BUCKET_NAME", None)
        s3 = _s3_client
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='uploads/')
        if 'Contents' not in response or not response['Contents']:
            return None
        latest_obj = max(response['Contents'], key=lambda x: x['LastModified'])
        return latest_obj['Key']
    except Exception as e:
        print(f"Error getting latest CSV from S3: {e}")
        return None

# Helper: convert datetime strings if needed
def _to_datetime_safe(s: str):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

# ------------------------------
# API Endpoints
# ------------------------------
@router.post("/start-training", response_model=TrainResponse, summary="Train models on uploaded CSV")
def train_pipeline():
    """
    Runs full preprocessing, feature engineering and trains XGB and LGB, selects best model.
    Persists a TrainingRun record (processing -> completed/failed).
    Uploads model & meta to S3 and saves S3 URIs in DB.
    Returns metrics and sample plot serialized as Plotly JSON.
    """
    global DATA_DF, LABEL_ENCODERS, FEATURES, BEST_MODEL, MODEL_TYPE, BEST_MODEL_META
    DATA_DF = get_csv_data()
    db = SessionLocal()
    training_run = None
    try:
        print('starting training pipeline...')
        if DATA_DF is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload-csv first.")

        # Create a training_run record (status = processing)
        upload_id = None
        try:
            csv_key = get_latest_csv_key()
            if csv_key:
                upload_rec = crud.get_upload_by_key(db, key=csv_key)
                if upload_rec:
                    upload_id = upload_rec.id
        except Exception:
            upload_id = None

        training_run = crud.create_training_run(db, upload_id=upload_id, user_id=None, status="processing")

        # Preprocess and train
        df_proc, LABEL_ENCODERS, FEATURES, TARGET = preprocess_and_feature_engineer(DATA_DF)
        best_model, model_type, metrics = train_models_and_select(df_proc, FEATURES, TARGET)
        print(f"Best model: {model_type} with metrics: {metrics}")

        # Name model files from the csv (or timestamp fallback)
        csv_key = get_latest_csv_key()
        if csv_key is None:
            csv_filename = f"local_upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        else:
            csv_filename = os.path.basename(csv_key)

        model_filename = f"{os.path.splitext(csv_filename)[0]}_model.pkl"
        meta_filename = f"{os.path.splitext(csv_filename)[0]}_meta.pkl"
        local_model_path = os.path.join(MODELS_DIR, model_filename)
        local_meta_path = os.path.join(MODELS_DIR, meta_filename)

        # Save local copies
        joblib.dump(best_model, local_model_path)
        joblib.dump({"model_type": model_type, "features": FEATURES, "label_encoders": LABEL_ENCODERS}, local_meta_path)

        # Upload model and meta to S3 under models/
        s3_model_key = f"models/{model_filename}"
        s3_meta_key = f"models/{meta_filename}"
        s3_upload_error = None
        s3_model_uri = None
        s3_meta_uri = None
        try:
            s3_model_uri = upload_file_to_s3(local_model_path, s3_model_key)
            s3_meta_uri = upload_file_to_s3(local_meta_path, s3_meta_key)
        except Exception as e:
            s3_upload_error = str(e)
            print(f"Error uploading model/meta to S3: {e}")

        BEST_MODEL = best_model
        MODEL_TYPE = model_type
        BEST_MODEL_META = {"model_type": model_type, "features": FEATURES, "label_encoders": LABEL_ENCODERS}

        # create a sample plot: last N actual vs predicted using test set
        df_proc = df_proc.sort_values('date').reset_index(drop=True)
        split_date = df_proc['date'].quantile(0.8)
        test_df = df_proc[df_proc['date'] >= split_date]
        X_test = test_df[FEATURES].drop(columns=['date'], errors='ignore')
        y_test = test_df[TARGET]
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

        # Update training_run as completed with metrics and S3/local paths
        crud.update_training_run(
            db,
            training_run.id,
            status="completed",
            metrics=metrics,
            model_path=s3_model_uri or local_model_path,
            model_meta_path=s3_meta_uri or local_meta_path,
            finished_at=datetime.utcnow(),
            rows_trained=int(df_proc.shape[0])
        )

        return {"message": "Training complete", "model_type": model_type, "metrics": metrics, "sample_fig_json": fig_dict, "training_run_id": training_run.id}

    except Exception as e:
        # mark training as failed
        err_msg = str(e)
        print("Training pipeline error:", err_msg)
        if training_run is not None:
            try:
                crud.fail_training_run(db, training_run.id, error_message=err_msg)
            except Exception as ee:
                print("Failed to update training_run status:", ee)
        raise HTTPException(status_code=500, detail=f"Training failed: {err_msg}")
    finally:
        db.close()


@router.post("/predict", summary="Predict future quantity for a product")
def predict(req: PredictRequest):
    """
    Predict future num_days for the given product (product_category, product, city).
    If the model is not loaded in memory, download the latest completed model & meta from S3 (using the latest TrainingRun)
    and load them. Persistes forecast in DB via save_forecast.
    Returns a JSON with predictions and a Plotly figure JSON (history + future).
    """
    global DATA_DF, BEST_MODEL, MODEL_TYPE, LABEL_ENCODERS, FEATURES, BEST_MODEL_META

    db = SessionLocal()
    try:
        # If model not loaded in memory, try to load from the latest completed training run (S3/local)
        if BEST_MODEL is None:
            runs = crud.list_training_runs(db, status="completed", limit=1, offset=0)
            if not runs:
                raise HTTPException(status_code=400, detail="No completed training run found. Call /start-training first.")
            latest = runs[0]
            model_path_uri = latest.model_path
            meta_path_uri = latest.model_meta_path

            # If paths are S3 URIs, download to tempfile; otherwise assume local file path
            loaded_model = None
            loaded_meta = None
            try:
                if model_path_uri and str(model_path_uri).startswith("s3://"):
                    tmp_model_path = download_s3_to_tempfile(model_path_uri)
                    loaded_model = joblib.load(tmp_model_path)
                    os.unlink(tmp_model_path)
                else:
                    loaded_model = joblib.load(model_path_uri)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model from {model_path_uri}: {e}")

            try:
                if meta_path_uri and str(meta_path_uri).startswith("s3://"):
                    tmp_meta_path = download_s3_to_tempfile(meta_path_uri)
                    loaded_meta = joblib.load(tmp_meta_path)
                    os.unlink(tmp_meta_path)
                else:
                    loaded_meta = joblib.load(meta_path_uri)
            except Exception as e:
                # Meta is not strictly required but helpful; warn rather than fail
                print(f"Warning: failed to load meta from {meta_path_uri}: {e}")
                loaded_meta = {}

            BEST_MODEL = loaded_model
            if isinstance(loaded_meta, dict):
                BEST_MODEL_META = loaded_meta
                MODEL_TYPE = loaded_meta.get("model_type", MODEL_TYPE)
                FEATURES = loaded_meta.get("features", FEATURES)
                LABEL_ENCODERS = loaded_meta.get("label_encoders", LABEL_ENCODERS)
            else:
                # fallback: leave MODEL_TYPE and others unchanged
                BEST_MODEL_META = {}

        if DATA_DF is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded. Use /upload-csv first.")

        # prepare base processed df & encoders/features if needed
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_plot['date'], y=hist_plot['quantity_sold'], name='historical_quantity_sold'))
        fig.add_trace(go.Scatter(x=future_preds['date'], y=future_preds['predicted_quantity_sold'], name='predicted_future'))
        fig.update_layout(title=f"History & {req.num_days}-day Forecast for {req.product} in {req.city}")
        figure_json = fig.to_json()

        # feature importance (if available)
        fi_json = None
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
        except Exception:
            fi_json = None

        # prepare predictions payload
        future_preds['date'] = future_preds['date'].astype(str)
        preds_table = future_preds[['date', 'predicted_quantity_sold']].to_dict(orient='records')

        # persist forecast to DB
        try:
            # try to find upload record from S3 latest CSV key
            upload_id = None
            try:
                csv_key = get_latest_csv_key()
                if csv_key:
                    upload_record = crud.get_upload_by_key(db, key=csv_key)
                    if upload_record:
                        upload_id = upload_record.id
            except Exception:
                upload_id = None

            # start & end dates
            start_date = _to_datetime_safe(preds_table[0]['date'])
            end_date = _to_datetime_safe(preds_table[-1]['date'])

            rec = crud.save_forecast(
                db,
                upload_id=upload_id,
                user_id=None,
                product_category=req.product_category,
                product=req.product,
                city=req.city,
                start_date=start_date,
                end_date=end_date,
                num_days=req.num_days,
                predictions=preds_table,
                params={"price": req.price, "discount": req.discount},
                figure_json=figure_json,
                feature_importance_json=fi_json
            )
            forecast_id = rec.id
        except Exception as e:
            # log but do not fail prediction result delivery
            print("Failed to persist forecast:", e)
            forecast_id = None

        return JSONResponse({
            "product": req.product,
            "city": req.city,
            "num_days": req.num_days,
            "predictions": preds_table,
            "figure_json": figure_json,
            "feature_importance_json": fi_json,
            "forecast_id": forecast_id
        })
    finally:
        db.close()


@router.get("/model/status")
def model_status():
    global BEST_MODEL, MODEL_TYPE
    return {"model_loaded": BEST_MODEL is not None, "model_type": MODEL_TYPE}
