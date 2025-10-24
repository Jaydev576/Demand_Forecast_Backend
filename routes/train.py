import os
import boto3
import json
import tempfile
from typing import Optional
from urllib.parse import urlparse
from datetime import datetime

from fastapi.params import Depends
import joblib
import pandas as pd

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import plotly.express as px
import plotly.graph_objects as go
from fastapi import APIRouter, BackgroundTasks

from sqlalchemy.orm import Session
from auth import get_current_user
from models import User
from schemas import PredictRequest
from routes.insights import generate_business_insight_background
from utils import generate_future_features, get_csv_data, preprocess_and_feature_engineer, sequential_predict, train_models_and_select, extract_and_store_features

# DB imports - adjust module path if different
from db import get_db, SessionLocal
import crud
from settings import settings

router = APIRouter()

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

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

# Helper to get CSV name from S3 uploads folder
def get_latest_csv_key(db: Session, user_id: int) -> Optional[str]:
    try:
        uploads = crud.list_uploads_for_user(db, user_id=user_id, limit=1)
        if uploads:
            return uploads[0].key
        return None
    except Exception as e:
        print(f"Error getting latest CSV from S3 for user {user_id}: {e}")
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
def train_pipeline(upload_id: int, db: Session, background_tasks: BackgroundTasks):
    """
    Runs full preprocessing, feature engineering and trains XGB and LGB, selects best model.
    Persists a TrainingRun record (processing -> completed/failed).
    Uploads model & meta to S3 and saves S3 URIs in DB.
    Returns metrics and sample plot serialized as Plotly JSON.
    """
    print(upload_id)
    
    db_session = db
    if db_session is None:
        db_session = SessionLocal()

    try:
        data_df = get_csv_data(upload_id, db_session)
        background_tasks.add_task(generate_business_insight_background, upload_id, db_session)
        #
        print("CSV data loaded from s3...", data_df.shape if data_df is not None else "None")
    except Exception as e:
        print(f"Error loading CSV from S3: {e}")
        data_df = None

    training_run = None
    try:
        print('Starting training pipeline...')
        if data_df is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded!")

        # Create a training_run record (status = processing)
        user_id = crud.get_userid_by_upload(db_session, upload_id)
        if not user_id:
            raise Exception(f"User not found for upload_id {upload_id}")

        extract_and_store_features(data_df, user_id, db_session)

        training_run = crud.create_training_run(db_session, upload_id=upload_id, user_id=user_id, status="processing")

        # Preprocess and train
        df_proc, label_encoders, features, target = preprocess_and_feature_engineer(data_df)
        best_model, model_type, metrics = train_models_and_select(df_proc, features, target)
        print(f"Best model: {model_type} with metrics: {metrics}")

        # Name model files from the csv (or timestamp fallback)
        csv_key = crud.get_upload_s3_key(db_session, upload_id)
        if csv_key is None:
            csv_filename = f"local_upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        else:
            csv_filename = os.path.basename(csv_key)

        model_filename = f"{os.path.splitext(csv_filename)[0]}_model.pkl"
        meta_filename = f"{os.path.splitext(csv_filename)[0]}_meta.joblib"
        local_model_path = os.path.join(MODELS_DIR, model_filename)
        local_meta_path = os.path.join(MODELS_DIR, meta_filename)

        # Save local copies
        joblib.dump(best_model, local_model_path)
        joblib.dump({"model_type": model_type, "features": features, "label_encoders": label_encoders}, local_meta_path)

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

        # create a sample plot: last N actual vs predicted using test set
        df_proc = df_proc.sort_values('date').reset_index(drop=True)
        split_date = df_proc['date'].quantile(0.8)
        test_df = df_proc[df_proc['date'] >= split_date]
        X_test = test_df[features].drop(columns=['date'], errors='ignore')
        y_test = test_df[target]
        preds = best_model.predict(X_test)

        fig = go.Figure()
        # plot actual mean per day (aggregate to reduce clutter)
        agg_actual = test_df.groupby('date')[target].sum().reset_index()
        agg_pred = pd.DataFrame({'date': test_df['date'], 'pred': preds}).groupby('date')['pred'].sum().reset_index()
        fig.add_trace(go.Scatter(x=agg_actual['date'], y=agg_actual[target], name='actual (sum/day)'))
        fig.add_trace(go.Scatter(x=agg_pred['date'], y=agg_pred['pred'], name='predicted (sum/day)'))
        fig.update_layout(title="Test Actual vs Predicted (daily aggregated)")
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)

        # Update training_run as completed with metrics and S3/local paths
        crud.update_training_run(
            db_session,
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
        print("Training pipeline error: ", err_msg)
        if training_run is not None:
            try:
                crud.fail_training_run(db_session, training_run.id, error_message=err_msg)
            except Exception as ee:
                print("Failed to update training_run status: ", ee)
        raise HTTPException(status_code=500, detail=f"Training failed: {err_msg}")
    finally:
        if db is None and db_session is not None:
            db_session.close()


@router.post("/predict", summary="Predict future quantity for a product")
def predict(req: PredictRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Predict future num_days for the given product (product_category, product, city).
    If the model is not loaded in memory, download the latest completed model & meta from S3 (using the latest TrainingRun)
    and load them. Persistes forecast in DB via save_forecast.
    Returns a JSON with predictions and a Plotly figure JSON (history + future).
    """
    try:
        # If model not loaded in memory, try to load from the latest completed training run (S3/local)
        runs = crud.list_training_runs(db, user_id=user.id, status="completed", limit=1, offset=0)
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

        model_type = loaded_meta.get("model_type", "xgb")
        features = loaded_meta.get("features", [])
        label_encoders = loaded_meta.get("label_encoders", {})

        upload_id = crud.list_uploads_for_user(db, user_id=user.id, limit=1)[0].id
        print(upload_id)
        data_df = get_csv_data(upload_id, db)
        if data_df is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload CSV first.")

        # prepare base processed df & encoders/features if needed
        base_df, label_encoders, features, target = preprocess_and_feature_engineer(data_df)

        # prepare future features
        future_df, hist_df, features = generate_future_features(
            req.product_category, req.product, req.city, num_days=req.num_days,
            base_df=base_df,
            price=req.price, discount=req.discount,
            label_encoders=label_encoders,
            features=features
        )

        # sequential predict
        future_preds = sequential_predict(loaded_model, model_type, future_df, hist_df, features)

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
            if model_type == 'xgb':
                importance = loaded_model.get_booster().get_score(importance_type='weight')
                fi = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
            else:
                if hasattr(loaded_model, 'feature_importances_'):
                    fi = pd.DataFrame({'feature': features, 'importance': loaded_model.feature_importances_}).sort_values('importance', ascending=False)
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
                csv_key = get_latest_csv_key(db, user.id)
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
                user_id=user.id,
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
            print("Failed to persist forecast: ", e)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")