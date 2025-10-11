# Forecast-AI

Forecast-AI is a powerful and easy-to-use web application for sales forecasting. It leverages machine learning to provide accurate predictions based on your historical sales data.

## Description

This project is a FastAPI-based web service that allows users to:
- Create an account and securely log in.
- Upload their sales data in CSV format.
- Train a machine learning model (XGBoost or LightGBM) on the uploaded data.
- Receive sales forecasts and visualize the results.

The application is designed to be robust and scalable, using modern technologies like Docker, AWS S3, and a PostgreSQL database.

## Features

- **User Authentication:** Secure user registration and login with JWT-based authentication and email verification.
- **File Uploads:** Upload sales data directly to a secure AWS S3 bucket.
- **Model Training:** Automatically train and select the best-performing model from XGBoost and LightGBM.
- **Sales Forecasting:** Predict future sales for specific products and locations.
- **Data Visualization:** View historical and forecasted sales data through interactive plots.
- **RESTful API:** A well-documented and easy-to-use API built with FastAPI.

## Tech Stack

- **Backend:** Python, FastAPI
- **Database:** PostgreSQL
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Pandas, NumPy
- **Authentication:** Python-JOSE (JWT), Passlib (for password hashing)
- **File Storage:** AWS S3
- **Email Service:** FastAPI-Mail
- **Containerization:** Docker

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.10+
- An AWS account with an S3 bucket
- A PostgreSQL database

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your_username/Forecast-AI.git
   cd Forecast-AI
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the root directory and add the following, replacing the placeholder values with your actual credentials:
   ```env
   DATABASE_URL="postgresql://user:password@host:port/database"
   SECRET_KEY="your_super_secret_key"
   ALGORITHM="HS256"
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   MAIL_USERNAME="your_email@example.com"
   MAIL_PASSWORD="your_email_password"
   MAIL_FROM="your_email@example.com"
   MAIL_PORT=587
   MAIL_SERVER="smtp.example.com"
   MAIL_STARTTLS=True
   MAIL_SSL_TLS=False
   AWS_ACCESS_KEY_ID="your_aws_access_key"
   AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
   AWS_REGION="your_aws_region"
   S3_BUCKET_NAME="your_s3_bucket_name"
   ```

5. **Run the application:**
   ```sh
   uvicorn main:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.

## API Endpoints

Here are some of the main API endpoints:

- `POST /user/signup`: Register a new user.
- `POST /user/login`: Log in and receive a JWT token.
- `GET /user/me`: Get the current user's details.
- `POST /upload/csv-for-training`: Upload a CSV file with sales data.
- `POST /train/start-training`: Initiate the model training process.
- `POST /train/predict`: Get sales predictions for a product.

For a complete list of endpoints and their details, see the auto-generated FastAPI documentation at `http://127.0.0.1:8000/docs`.

## Project Structure

```
Forecast-AI/
├── .env                # Environment variables
├── .gitignore          # Git ignore file
├── auth.py             # Authentication logic
├── crud.py             # Database CRUD operations
├── db.py               # Database session management
├── docker-compose.yml  # Docker Compose configuration
├── main.py             # FastAPI application entry point
├── models.py           # SQLAlchemy database models
├── requirements.txt    # Python dependencies
├── schemas.py          # Pydantic data validation schemas
├── security.py         # Security-related utilities
├── settings.py         # Application settings
├── utils.py            # Utility functions
├── routes/             # API route definitions
│   ├── train.py
│   ├── uploads.py
│   └── users.py
└── ...
```

---

Happy Forecasting!
