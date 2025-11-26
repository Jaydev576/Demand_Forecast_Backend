import os
import boto3
import json
from datetime import datetime
import joblib
import pandas as pd
from fastapi import HTTPException
import plotly.graph_objects as go
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from utils.email import send_training_completion_email
from routers.insights import generate_business_insight_background
from utils.helpers import get_csv_data, preprocess_and_feature_engineer, train_models_and_select, extract_and_store_features
from db.db import SessionLocal
import services.crud as crud
from core.config import settings

# Ensure cache/models/ directory exists for local model storage
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache/models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# S3 client (uses credentials from settings / env)
_s3_client = boto3.client(
    "s3",
    region_name=getattr(settings, "AWS_REGION", None),
    aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
    aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
)

# -------------------- Helpers --------------------
def upload_file_to_s3(local_path: str, s3_key: str):
    """ Upload local file to configured S3 bucket under s3_key (uploads/ or models/). """
    bucket = getattr(settings, "S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3 bucket not configured in settings.S3_BUCKET_NAME")
    _s3_client.upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


# Training pipeline
def train_pipeline(upload_id: int, db: Session, background_tasks: BackgroundTasks):
    """
    Runs full preprocessing, feature engineering and trains XGB and LGB, selects best model.
    Persists a TrainingRun record (processing -> completed/failed).
    Uploads model & meta to S3 and saves S3 URIs in DB.
    Returns metrics and sample plot serialized as Plotly JSON.
    """
    print("uploaded id : ", upload_id)
    
    db_session = db
    if db_session is None:
        db_session = SessionLocal()

    try:
        data_df = get_csv_data(upload_id, db_session)
        background_tasks.add_task(generate_business_insight_background, upload_id, db_session)
        # print("CSV data loaded from s3...", data_df.shape if data_df is not None else "None")
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
        
        user = crud.get_user(db_session, user_id)
        if not user:
            raise Exception(f"User not found for user_id {user_id}")

        # Extract and store features
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

        # # plot actual mean per day (aggregate to reduce clutter)
        # fig = go.Figure()
        # agg_actual = test_df.groupby('date')[target].sum().reset_index()
        # agg_pred = pd.DataFrame({'date': test_df['date'], 'pred': preds}).groupby('date')['pred'].sum().reset_index()
        # fig.add_trace(go.Scatter(x=agg_actual['date'], y=agg_actual[target], name='actual (sum/day)'))
        # fig.add_trace(go.Scatter(x=agg_pred['date'], y=agg_pred['pred'], name='predicted (sum/day)'))
        # fig.update_layout(title="Test Actual vs Predicted (daily aggregated)")
        # fig_json = fig.to_json()
        # fig_dict = json.loads(fig_json)

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
        background_tasks.add_task(send_training_completion_email, [user.email], user.username)
        print("Training pipeline completed successfully.")
        
    except Exception as e:
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