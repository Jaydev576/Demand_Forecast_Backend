import uuid
import boto3
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session
from botocore.exceptions import ClientError
from auth import get_current_active_user
from db import get_db
from routes.train import train_pipeline
from settings import settings
from models import Upload, User
from schemas import UploadCompleteRequest
from typing import Optional

router = APIRouter()

# initialize S3 client from settings (ensure credentials are set)
s3 = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)

def make_s3_key(filename: str) -> str:
    """
    Create a unique S3 key for uploads.
    Keeps extension if present, defaults to csv.
    """
    # sanitize base filename (remove path segments, limit length)
    base = filename.split("/")[-1].split("\\")[-1]
    ext = base.split(".")[-1].lower() if "." in base else "csv"
    ext = ext if len(ext) <= 8 else "csv"
    return f"uploads/{uuid.uuid4().hex}.{ext}"

# Optional auth dependency; replace with your project's current_user dependency if available
# def get_current_user_optional():
#     # stub: replace with actual auth dependency or remove from route signature
#     return None

# @router.post("/csv-for-training", status_code=status.HTTP_201_CREATED)
# async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user_optional)):
#     """
#     Uploads the given file directly to S3, and inserts a DB record in `uploads`.
#     Returns upload metadata (db id + s3 key).
#     """
#     key = make_s3_key(file.filename)
#     bucket_name = settings.S3_BUCKET_NAME

#     # rewind file if needed
#     try:
#         file.file.seek(0)
#     except Exception:
#         pass

#     try:
#         s3.upload_fileobj(
#             Fileobj=file.file,
#             Bucket=bucket_name,
#             Key=key,
#             ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
#         )
#     except ClientError as e:
#         raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

#     # get object metadata
#     try:
#         head = s3.head_object(Bucket=bucket_name, Key=key)
#         size = head.get("ContentLength", 0)
#     except ClientError as e:
#         # if head fails, we still record entry but log/return an error
#         size = 0
#         # optional: delete the uploaded object to avoid orphaned objects
#         try:
#             s3.delete_object(Bucket=bucket_name, Key=key)
#         except Exception:
#             pass
#         raise HTTPException(status_code=500, detail=f"S3 head_object failed after upload: {e}")

#     # --- insert s3 metadata into db ---
#     try:
#         user_id = current_user.id if current_user is not None else None
#         row = Upload(
#             user_id=user_id,
#             filename=file.filename,
#             key=key,
#             bucket=bucket_name,
#             size_bytes=int(size),
#             content_type=file.content_type or "application/octet-stream",
#         )
#         db.add(row)
#         db.commit()
#         db.refresh(row)
#     except Exception as e:
#         db.rollback()
#         # optionally delete uploaded S3 object to avoid orphaned objects
#         try:
#             s3.delete_object(Bucket=bucket_name, Key=key)
#         except Exception:
#             pass
#         raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

#     return {
#         "id": row.id,
#         "filename": row.filename,
#         "s3_key": row.key,
#         "bucket": row.bucket,
#         "size_bytes": row.size_bytes,
#         "content_type": row.content_type,
#     }


@router.get("/generate-upload-url")
def generate_presigned_upload_url(
    filename: str = Query(...),
    content_type: str = Query("text/csv"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_active_user),
):
    """
    Generates a presigned PUT URL that the client can use to upload a file directly to S3.
    Creates a DB Upload record (size_bytes=0 for now). Caller should call /upload-complete after client uploads.
    """
    bucket_name = settings.S3_BUCKET_NAME
    key = make_s3_key(filename)

    try:
        presigned_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket_name,
                "Key": key,
                "ContentType": content_type
            },
            ExpiresIn=300  # 5 minutes
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {e}")

    # create DB record (size will be updated in /upload-complete)
    try:
        user_id = current_user.id if current_user is not None else None
        row = Upload(
            user_id=user_id,
            filename=filename,
            key=key,
            bucket=bucket_name,
            size_bytes=0,
            content_type=content_type,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    return {
        "upload_url": presigned_url,
        "key": key,
        "upload_id": row.id
    }


@router.post("/upload-complete")
def confirm_upload(
    request: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Called by client after uploading with presigned URL.
    Updates DB record with size and metadata from S3.
    """
    upload = db.get(Upload, request.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload record not found")

    try:
        head = s3.head_object(Bucket=upload.bucket, Key=upload.key)
        upload.size_bytes = int(head.get("ContentLength", 0))
        # optionally update content_type from head if needed
        upload.content_type = head.get("ContentType", upload.content_type)
        db.add(upload)
        db.commit()
        db.refresh(upload)

        # start training in background
        background_tasks.add_task(train_pipeline, upload.id, db, background_tasks)
        # print("upload: ", upload.id)
        return {"status": "ok", "upload_id": upload.id, "size_bytes": upload.size_bytes}

    except ClientError as e:
        # do not overwrite DB if head fails; surface error
        raise HTTPException(status_code=500, detail=f"S3 head_object failed: {e}")


@router.get("/download/{upload_id}")
def get_download_url(upload_id: int, db: Session = Depends(get_db)):
    """
    Generate a short-lived presigned GET URL for a previously uploaded file.
    """
    row = db.get(Upload, upload_id)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")

    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": row.bucket, "Key": row.key},
            ExpiresIn=300,  # 5 minutes
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned GET URL: {e}")

    return {"url": url}
