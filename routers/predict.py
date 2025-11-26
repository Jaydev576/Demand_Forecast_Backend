import os
import datetime
import tempfile
from typing import Optional
from urllib.parse import urlparse
import boto3
import joblib
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from fastapi.params import Depends
from fastapi.responses import JSONResponse
# import plotly.express as px

from utils.auth import get_current_user
from models.models import User
from schemas.schemas import PredictRequest
import core.config as config
from utils.helpers import generate_future_features, get_csv_data, preprocess_and_feature_engineer, sequential_predict
import services.crud as crud
from db.db import get_db

router = APIRouter()

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache/models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# S3 client (uses credentials from settings / env)
_s3_client = boto3.client(
    "s3",
    region_name=getattr(config, "AWS_REGION", None),
    aws_access_key_id=getattr(config, "AWS_ACCESS_KEY_ID", None),
    aws_secret_access_key=getattr(config, "AWS_SECRET_ACCESS_KEY", None),
)

# --------- Helpers ---------
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

# convert datetime strings if needed
def _to_datetime_safe(s: str):
    try:
        
        return datetime.datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None


# --------- Routes ---------
@router.post("/", summary="Predict future quantity for a product")
def predict(req: PredictRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Predict future num_days for the given product (product_category, product, city).
    If the model is not loaded in memory, download the latest completed model & meta from S3 (using the latest TrainingRun)
    and load them. Persistes forecast in DB via save_forecast.
    Returns a JSON with predictions and a Plotly figure JSON (history + future).
    """
    try:
        # If model not loaded in memory, try to load from the latest completed training run (S3/local)
        runs = crud.list_training_runs(db, user_id=user.id, limit=1, offset=0)
        if not runs:
            raise HTTPException(status_code=400, detail="CSV dataset not found. Please upload CSV and train model first.")
        latest = runs[0]
        if latest.status == "processing":
            raise HTTPException(status_code=400, detail="Model training is still in progress. Please try again later.")
        model_path_uri = latest.model_path
        meta_path_uri = latest.model_meta_path

        # If paths are S3 URIs, download to tempfile; otherwise assume local file path
        loaded_model = None
        loaded_meta = None

        # Construct local paths
        model_filename = os.path.basename(model_path_uri) if model_path_uri else ""
        meta_filename = os.path.basename(meta_path_uri) if meta_path_uri else ""
        local_model_path = os.path.join(MODELS_DIR, model_filename)
        local_meta_path = os.path.join(MODELS_DIR, meta_filename)

        try:
            if os.path.exists(local_model_path):
                loaded_model = joblib.load(local_model_path)
            elif model_path_uri and str(model_path_uri).startswith("s3://"):
                tmp_model_path = download_s3_to_tempfile(model_path_uri)
                loaded_model = joblib.load(tmp_model_path)
                os.unlink(tmp_model_path)
            else:
                raise HTTPException(status_code=500, detail=f"Model not found locally or on S3: {model_path_uri}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model from {model_path_uri}: {e}")

        try:
            if os.path.exists(local_meta_path):
                loaded_meta = joblib.load(local_meta_path)
            elif meta_path_uri and str(meta_path_uri).startswith("s3://"):
                tmp_meta_path = download_s3_to_tempfile(meta_path_uri)
                loaded_meta = joblib.load(tmp_meta_path)
                os.unlink(tmp_meta_path)
                print("meta loaded successfully from S3.")
            else:
                # Meta is not strictly required but helpful; warn rather than fail
                print(f"Warning: meta not found locally or on S3: {meta_path_uri}")
                loaded_meta = {}
        except Exception as e:
            # Meta is not strictly required but helpful; warn rather than fail
            print(f"Warning: failed to load meta from {meta_path_uri}: {e}")
            loaded_meta = {}

        model_type = loaded_meta.get("model_type", "xgb")
        features = loaded_meta.get("features", [])
        label_encoders = loaded_meta.get("label_encoders", {})

        upload_id = crud.list_uploads_for_user(db, user_id=user.id, limit=1)[0].id
        print("upload id:",upload_id)
        data_df = get_csv_data(upload_id, db)
        if data_df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataset for prediction.")
        print("dataframe loaded...")
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
        future_preds = sequential_predict(loaded_model, future_df, hist_df, features)

        # # build a plotly graph: history (last 90 days) + future
        # hist_plot = hist_df.sort_values('date').tail(90).copy()
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=hist_plot['date'], y=hist_plot['quantity_sold'], name='historical_quantity_sold'))
        # fig.add_trace(go.Scatter(x=future_preds['date'], y=future_preds['predicted_quantity_sold'], name='predicted_future'))
        # fig.update_layout(title=f"History & {req.num_days}-day Forecast for {req.product} in {req.city}")
        # figure_json = fig.to_json()

        # # feature importance (if available)
        # fi_json = None
        # try:
        #     if model_type == 'xgb':
        #         importance = loaded_model.get_booster().get_score(importance_type='weight')
        #         fi = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
        #     else:
        #         if hasattr(loaded_model, 'feature_importances_'):
        #             fi = pd.DataFrame({'feature': features, 'importance': loaded_model.feature_importances_}).sort_values('importance', ascending=False)
        #     if fi is not None and not fi.empty:
        #         fig2 = px.bar(fi.head(20), x='feature', y='importance', title='Top 20 Feature Importances')
        #         fi_json = fig2.to_json()
        # except Exception:
        #     fi_json = None

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
                # figure_json=figure_json,
                # feature_importance_json=fi_json
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
            # "figure_json": figure_json,
            # "feature_importance_json": fi_json,
            "forecast_id": forecast_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    

@router.get("/get_forecasts", summary="Get list of forecasts for the current user")
def get_forecasts(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Returns a list of forecasts made by the current user.
    """
    try:
        forecasts = crud.list_forecasts_for_user(db, user_id=user.id)
        if not forecasts:
            return JSONResponse({"forecasts": []})
        
        return JSONResponse({"forecasts": [f.to_dict() for f in forecasts]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")