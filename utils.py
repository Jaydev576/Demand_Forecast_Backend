# utils.py
from typing import List, Optional, Tuple, Dict
import boto3
from fastapi import HTTPException
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from settings import settings

# Globals (updated by preprocess)
LABEL_ENCODERS: Dict[str, LabelEncoder] = {}
FEATURES: List[str] = []
TARGET = "quantity_sold"

# allowed categorical columns we will encode
CATEGORICAL_COLS = ['product_category', 'product', 'city', 'season']

# ---------------------------
# S3 helpers for CSV loading
# ---------------------------
def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=getattr(settings, "AWS_REGION", None),
        aws_access_key_id=getattr(settings, "AWS_ACCESS_KEY_ID", None),
        aws_secret_access_key=getattr(settings, "AWS_SECRET_ACCESS_KEY", None),
    )

def get_csv_data() -> Optional[pd.DataFrame]:
    """
    Loads the CSV data from S3 (the latest uploaded one under uploads/).
    Returns a DataFrame or None if not found.
    """
    s3 = _get_s3_client()
    bucket_name = settings.S3_BUCKET_NAME

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='uploads/')
        if 'Contents' not in response or not response['Contents']:
            return None
        objects = response['Contents']
        latest_obj = max(objects, key=lambda x: x['LastModified'])
        key = latest_obj['Key']
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        # obj['Body'] is a StreamingBody; pass directly to pandas
        df = pd.read_csv(obj['Body'])
        # ensure date columns parsed lazily later in preprocess
        return df
    except Exception as e:
        print(f"Error loading CSV from S3: {e}")
        return None

# ---------------------------
# Preprocessing & features
# ---------------------------
def _safe_to_datetime(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        df[col] = pd.NaT
    return df

def preprocess_and_feature_engineer(df: pd.DataFrame):
    """
    Accepts raw df and returns:
      df_processed, LABEL_ENCODERS, FEATURES, target (string)
    Mutates global LABEL_ENCODERS and FEATURES.
    """
    global LABEL_ENCODERS, FEATURES, TARGET

    df = df.copy()

    # holiday indicator (existing column or later filled)
    df['holiday'] = df.get('holiday', pd.NA).apply(lambda x: 0 if pd.isna(x) else 1)

    # date columns
    df = _safe_to_datetime(df, 'date')
    df = _safe_to_datetime(df, 'release_date')

    # sort
    df = df.sort_values(by=['product_category', 'product', 'city', 'date']).reset_index(drop=True)

    # date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # holidays: try python-holidays
    try:
        import holidays as _hol
        years = range(int(df['date'].dt.year.min()), int(df['date'].dt.year.max()) + 1)
        ind_holidays = _hol.India(years=years)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if x in ind_holidays else 0)
    except Exception:
        df['is_holiday'] = df['holiday'].fillna(0).astype(int)

    # label encode categoricals
    LABEL_ENCODERS = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # convert to str and fill missing
        df[col] = df.get(col, "").astype(str).fillna("NA")
        le.fit(df[col])
        df[col] = le.transform(df[col])
        LABEL_ENCODERS[col] = le

    # product age
    df['product_age_days'] = (df['date'] - df['release_date']).dt.days.fillna(0).astype(int)

    # lag & rolling features grouped by product
    for lag in [1, 7, 30]:
        df[f'quantity_sold_lag_{lag}'] = df.groupby('product')['quantity_sold'].shift(lag)

    for window in [7, 30]:
        df[f'quantity_sold_roll_mean_{window}'] = df.groupby('product')['quantity_sold'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'quantity_sold_roll_std_{window}'] = df.groupby('product')['quantity_sold'].transform(lambda x: x.rolling(window=window, min_periods=1).std(ddof=0))

    # Fill NaNs
    df.fillna(0, inplace=True)

    # FEATURES list
    FEATURES = ['product_category', 'product', 'city', 'year', 'month', 'day', 'dayofweek', 'weekofyear',
                'quarter', 'is_weekend',
                'is_holiday', 'price', 'discount', 'final_price', 'competitor_price',
                'marketing_spend', 'last_month_sales', 'product_age_days',
                'quantity_sold_lag_1', 'quantity_sold_lag_7', 'quantity_sold_lag_30',
                'quantity_sold_roll_mean_7', 'quantity_sold_roll_std_7',
                'quantity_sold_roll_mean_30', 'quantity_sold_roll_std_30']

    # ensure features exist
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0

    TARGET = 'quantity_sold'
    return df, LABEL_ENCODERS, FEATURES, TARGET

# ---------------------------
# Training & model selection
# ---------------------------
def train_models_and_select(df: pd.DataFrame, features: List[str], target: str):
    """
    Train XGB model (and optionally LGB) and return best model, model_type, and metrics summary.
    """
    # prepare train/test by time split
    df = df.copy().sort_values('date').reset_index(drop=True)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    X_train = train_df[features].drop(columns=['date'], errors='ignore')
    y_train = train_df[target]
    X_test = test_df[features].drop(columns=['date'], errors='ignore')
    y_test = test_df[target]

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    print("Columns used:", X_train.columns.tolist())

    # XGBoost model (sklearn wrapper)
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective='reg:squarederror',
        verbosity=0,
        n_jobs=-1
    )

    # Try fit with callback (xgboost >= 2.0), else fallback
    try:
        from xgboost.callback import EarlyStopping
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False,
                      callbacks=[EarlyStopping(rounds=50, save_best=True)])
    except Exception:
        try:
            # fallback older API
            xgb_model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          verbose=False,
                          early_stopping_rounds=50)
        except Exception as e:
            # last resort: plain fit
            print("Warning: early stopping not available, fitting without early stopping:", e)
            xgb_model.fit(X_train, y_train, verbose=False)

    # Evaluate
    def eval_model(m, X, y):
        pred = m.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    xgb_metrics = eval_model(xgb_model, X_test, y_test)

    best_model = xgb_model
    model_type = 'xgb'
    best_metrics = xgb_metrics

    return best_model, model_type, {"xgb": xgb_metrics, "selected": best_metrics}

# ---------------------------
# Small helpers (season, holiday, discount)
# ---------------------------
def get_season(month: int):
    if month in [12, 1, 2]:
        return 'winter'
    if month in [3, 4, 5]:
        return 'spring'
    if month in [6, 7, 8]:
        return 'summer'
    return 'autumn'

def get_holiday(date: pd.Timestamp):
    try:
        import holidays as _hol
        ind_holidays = _hol.India(years=[date.year])
        return date in ind_holidays
    except Exception:
        return False

def get_discount(season, holiday):
    base = 0.05
    if season == 'winter':
        base += 0.02
    if holiday:
        base += 0.05
    return base

# ---------------------------
# Future generation & prediction
# ---------------------------
def generate_future_features(product_category: str, product: str, city: str, num_days: int = 30,
                             base_df: pd.DataFrame = None, price: Optional[float] = None,
                             discount: Optional[float] = None, seed: Optional[int] = None):
    """
    Build deterministic future features using LABEL_ENCODERS and base_df.
    Returns (future_df, prod_df, FEATURES)
    """
    global LABEL_ENCODERS, FEATURES

    if base_df is None:
        raise ValueError("base_df required")

    # ensure label encoders present
    if not LABEL_ENCODERS:
        # attempt to precompute encoders from base_df
        _, LABEL_ENCODERS, FEATURES, _ = preprocess_and_feature_engineer(base_df)

    # encode inputs (raise HTTP 400 if unseen)
    try:
        enc_category = LABEL_ENCODERS['product_category'].transform([str(product_category)])[0]
        enc_product = LABEL_ENCODERS['product'].transform([str(product)])[0]
        enc_city = LABEL_ENCODERS['city'].transform([str(city)])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {e}")

    prod_df = base_df[(base_df['product_category'] == enc_category) &
                      (base_df['product'] == enc_product) &
                      (base_df['city'] == enc_city)].copy()

    if prod_df.empty:
        raise HTTPException(status_code=404, detail="Product+city combination not found in dataset")

    prod_df = prod_df.sort_values('date').reset_index(drop=True)
    last_row = prod_df.iloc[-1]
    last_date = prod_df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days, freq='D')

    rows = []
    for date in future_dates:
        season = get_season(date.month)
        holiday = get_holiday(date)
        is_holiday = 1 if holiday else 0
        current_price = float(price) if price is not None else float(last_row.get('price', 0.0))
        current_discount = float(discount) if discount is not None else get_discount(season, holiday)
        final_price = current_price * (1 - current_discount)
        competitor_price = float(last_row.get('competitor_price', 0.0))
        marketing_spend = float(last_row.get('marketing_spend', 0.0))
        last_month_sales = float(last_row.get('last_month_sales', 0.0))
        product_age_days = (date - pd.to_datetime(last_row['release_date'])).days

        # lags fallback
        quantity_sold_lag_1 = float(last_row.get('quantity_sold', 0.0))
        quantity_sold_lag_7 = float(prod_df['quantity_sold'].shift(6).iloc[-1]) if len(prod_df) > 6 else quantity_sold_lag_1
        quantity_sold_lag_30 = float(prod_df['quantity_sold'].shift(29).iloc[-1]) if len(prod_df) > 29 else quantity_sold_lag_1
        q_mean_7 = float(last_row.get('quantity_sold_roll_mean_7', prod_df['quantity_sold'].tail(7).mean() if len(prod_df) >= 1 else 0.0))
        q_std_7 = float(last_row.get('quantity_sold_roll_std_7', prod_df['quantity_sold'].tail(7).std(ddof=0) if len(prod_df) >= 1 else 0.0))
        q_mean_30 = float(last_row.get('quantity_sold_roll_mean_30', prod_df['quantity_sold'].tail(30).mean() if len(prod_df) >= 1 else 0.0))
        q_std_30 = float(last_row.get('quantity_sold_roll_std_30', prod_df['quantity_sold'].tail(30).std(ddof=0) if len(prod_df) >= 1 else 0.0))

        dayofweek = date.dayofweek
        weekofyear = date.isocalendar().week
        quarter = date.quarter
        is_weekend = 1 if dayofweek in [5,6] else 0

        rows.append({
            'product_category': enc_category,
            'product': enc_product,
            'city': enc_city,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'dayofweek': dayofweek,
            'weekofyear': weekofyear,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'price': current_price,
            'discount': current_discount,
            'final_price': final_price,
            'competitor_price': competitor_price,
            'marketing_spend': marketing_spend,
            'last_month_sales': last_month_sales,
            'product_age_days': product_age_days,
            'quantity_sold_lag_1': quantity_sold_lag_1,
            'quantity_sold_lag_7': quantity_sold_lag_7,
            'quantity_sold_lag_30': quantity_sold_lag_30,
            'quantity_sold_roll_mean_7': q_mean_7,
            'quantity_sold_roll_std_7': q_std_7,
            'quantity_sold_roll_mean_30': q_mean_30,
            'quantity_sold_roll_std_30': q_std_30,
            'date': date
        })

    future_df = pd.DataFrame(rows)
    # ensure FEATURES variable is present and does not include the target
    if not FEATURES:
        _, LABEL_ENCODERS, FEATURES, _ = preprocess_and_feature_engineer(base_df)
    # return features list as-is (without target)
    return future_df, prod_df, FEATURES

def sequential_predict(model, model_type: str, future_df: pd.DataFrame, historical_df: pd.DataFrame, features: List[str]):
    """
    Predict sequentially for future_df updating lags/rolling using predicted values (approx).
    Returns future_df with 'predicted_quantity_sold' column (rounded ints).
    """
    preds = []
    hist = historical_df.copy().reset_index(drop=True)

    # prepare rolling lists; if insufficient history, pad with last known or zeros
    rolling_7 = list(hist['quantity_sold'].tail(6)) if len(hist) >= 1 else [0]*6
    rolling_30 = list(hist['quantity_sold'].tail(29)) if len(hist) >= 1 else [0]*29

    # ensure features exist on each row
    for i in range(len(future_df)):
        row = future_df.iloc[i:i+1].copy()

        # set lags
        if i == 0:
            row.loc[:, 'quantity_sold_lag_1'] = hist.iloc[-1]['quantity_sold'] if len(hist) >= 1 else 0.0
            row.loc[:, 'quantity_sold_lag_7'] = hist['quantity_sold'].shift(6).iloc[-1] if len(hist) > 6 else row.loc[:, 'quantity_sold_lag_7']
            row.loc[:, 'quantity_sold_lag_30'] = hist['quantity_sold'].shift(29).iloc[-1] if len(hist) > 29 else row.loc[:, 'quantity_sold_lag_30']
        else:
            row.loc[:, 'quantity_sold_lag_1'] = preds[-1]
            if i >= 7:
                row.loc[:, 'quantity_sold_lag_7'] = preds[-7]
            if i >= 30:
                row.loc[:, 'quantity_sold_lag_30'] = preds[-30]

        # update rolling stats
        if i > 0:
            rolling_7.append(preds[-1])
            if len(rolling_7) > 7:
                rolling_7.pop(0)
            row.loc[:, 'quantity_sold_roll_mean_7'] = np.mean(rolling_7) if len(rolling_7) else 0.0
            row.loc[:, 'quantity_sold_roll_std_7'] = np.std(rolling_7, ddof=0) if len(rolling_7) else 0.0

            rolling_30.append(preds[-1])
            if len(rolling_30) > 30:
                rolling_30.pop(0)
            row.loc[:, 'quantity_sold_roll_mean_30'] = np.mean(rolling_30) if len(rolling_30) else 0.0
            row.loc[:, 'quantity_sold_roll_std_30'] = np.std(rolling_30, ddof=0) if len(rolling_30) else 0.0

        # ensure all features exist in row
        for f in features:
            if f not in row.columns:
                row.loc[:, f] = 0

        X = row[features].astype(float)

        # predict
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

        pred = max(0.0, float(pred))
        preds.append(pred)

    future_df = future_df.copy()
    future_df['predicted_quantity_sold'] = [int(round(p)) for p in preds]
    return future_df
