# import uuid
# import boto3
# from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
# from sqlalchemy.orm import Session
# from botocore.exceptions import ClientError

# import crud
# from db import get_db
# from settings import settings
# from models import Upload

# router = APIRouter()

# s3 = boto3.client(
#     "s3",
#     region_name=settings.AWS_REGION,
#     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
# )

# def make_s3_key(filename: str) -> str:
#     ext = filename.split(".")[-1].lower() if "." in filename else "csv"
#     return f"uploads/{uuid.uuid4().hex}.{ext}"


# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     key = make_s3_key(file.filename)
#     bucket_name = settings.S3_BUCKET_NAME

#     try:
#         await s3.upload_fileobj(
#             Fileobj=file.file,
#             Bucket=bucket_name,
#             Key=key,
#             ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
#         )
#     except ClientError as e:
#         raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

#     head = s3.head_object(Bucket=bucket_name, Key=key)
#     size = head.get("ContentLength")


#     # --- inserting s3 metadata into db ---

#     # row = Upload(
#     #     filename=file.filename,
#     #     key=key,
#     #     bucket=bucket_name,
#     #     size_bytes=size,
#     #     content_type=file.content_type or "application/octet-stream",
#     # )
#     # db.add(row)
#     # db.commit()
#     # db.refresh(row)

#     # return {
#     #     "id": row.id,
#     #     "filename": row.filename,
#     #     "s3_key": row.key,
#     #     "bucket": row.bucket,
#     #     "size_bytes": row.size_bytes,
#     #     "content_type": row.content_type,
#     # }


# # --- to generate a presigned GET URL of csv file ---
# @router.get("/download/{upload_id}")
# def get_download_url(upload_id: int, db: Session = Depends(get_db)):
#     row = db.get(Upload, upload_id)
#     if not row:
#         raise HTTPException(404, "Not found")
#     url = s3.generate_presigned_url(
#         ClientMethod="get_object",
#         Params={"Bucket": row.bucket, "Key": row.key},
#         ExpiresIn=300,  # 5 minutes
#     )
#     return {"url": url}
