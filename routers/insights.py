import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import services.crud as crud
from utils.auth import get_current_active_user
from db.db import get_db
from models.models import User
from utils.helpers import get_csv_data
import schemas.schemas as schemas

router = APIRouter()

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def generate_business_insight_background(upload_id: int, db: Session):
    """
    Generate and store business insights for a given upload.
    """
    user_id = crud.get_userid_by_upload(db, upload_id)
    if not user_id:
        print(f"No user found for upload_id {upload_id}")
        return

    data_df = get_csv_data(upload_id, db)
    if data_df is None:
        print(f"Could not load data for upload_id {upload_id}")
        return

    # Calculate KPIs
    kpis = {}
    kpis["total_sales"] = data_df["quantity_sold"].sum()
    kpis["total_revenue"] = (data_df["quantity_sold"] * data_df["price"]).sum()

    # Generate charts
    charts = {}
    # Top performing products, categories, and cities
    charts["top_products"] = data_df.groupby("product")["quantity_sold"].sum().sort_values(ascending=False).to_dict()
    charts["top_categories"] = data_df.groupby("product_category")["quantity_sold"].sum().sort_values(ascending=False).to_dict()
    charts["top_cities"] = data_df.groupby("city")["quantity_sold"].sum().sort_values(ascending=False).to_dict()

    # Product category-wise contribution of different products for pie chart
    category_product_contribution = {}
    for category in data_df["product_category"].unique():
        category_df = data_df[data_df["product_category"] == category]
        product_contribution = category_df.groupby("product")["quantity_sold"].sum().to_dict()
        category_product_contribution[category] = product_contribution
    charts["category_product_contribution"] = category_product_contribution

    # Create and store the new insight
    new_insight = schemas.BusinessInsightCreate(
        user_id=user_id,
        kpis=convert_numpy_types(kpis),
        charts=convert_numpy_types(charts),
    )
    crud.create_business_insight(db, insight=new_insight)
    print(f"Business insights generated for upload_id {upload_id} in background.")


@router.get("/")
def get_business_insights(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    insights = crud.get_latest_business_insight(db, user_id=current_user.id)
    if insights:
        return insights

    # If no insights found, generate them
    # Get the latest upload for the user
    latest_upload = crud.list_uploads_for_user(db, user_id=current_user.id, limit=1)
    if not latest_upload:
        raise HTTPException(status_code=404, detail="No data uploaded yet")

    # Get the data from the latest upload
    data_df = get_csv_data(latest_upload[0].id, db)
    if data_df is None:
        raise HTTPException(status_code=404, detail="Could not load data from the latest upload")

    # Calculate KPIs
    kpis = {}
    kpis["total_sales"] = data_df["quantity_sold"].sum()
    kpis["total_revenue"] = (data_df["quantity_sold"] * data_df["price"]).sum()

    # Generate charts
    charts = {}
    # Top performing products, categories, and cities
    charts["top_products"] = data_df.groupby("product")["quantity_sold"].sum().sort_values(ascending=False).to_dict()
    charts["top_categories"] = data_df.groupby("product_category")["quantity_sold"].sum().sort_values(ascending=False).to_dict()
    charts["top_cities"] = data_df.groupby("city")["quantity_sold"].sum().sort_values(ascending=False).to_dict()

    # Product category-wise contribution of different products for pie chart
    category_product_contribution = {}
    for category in data_df["product_category"].unique():
        category_df = data_df[data_df["product_category"] == category]
        product_contribution = category_df.groupby("product")["quantity_sold"].sum().to_dict()
        category_product_contribution[category] = product_contribution
    charts["category_product_contribution"] = category_product_contribution
    
    # Create and store the new insight
    new_insight = schemas.BusinessInsightCreate(
        user_id=current_user.id,
        kpis=convert_numpy_types(kpis),
        charts=convert_numpy_types(charts),
    )
    bi = crud.create_business_insight(db, insight=new_insight)
    return ({
        "kpis": bi.kpis,
        "charts": bi.charts,
    })
