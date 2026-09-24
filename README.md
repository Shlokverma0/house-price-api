# 🏠 House Price Prediction API

A production-grade **Machine Learning REST API** that predicts house prices based on property features and location. Built with **FastAPI** following a clean **4-Layer Architecture** (Routes → Controllers → Services → Repositories).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [Setup & Installation](#-setup--installation)
- [Running the API](#-running-the-api)
- [API Endpoints](#-api-endpoints)
- [Example Request & Response](#-example-request--response)
- [Running Tests](#-running-tests)
- [Author](#-author)

---

## 🎯 Overview

This project is a **House Price Prediction API** that:
- Uses a **real Kaggle dataset** of 250,000+ Indian house records.
- Trains a **Multiple Linear Regression** model on 47 features (5 base features + 42 location one-hot columns).
- Serves predictions via a clean, modular **FastAPI** backend.
- Follows industry-standard **4-layer separation of concerns**.

---

## 🏗 Architecture

This project follows a **4-Layer Architecture** (also known as Controller-Service-Repository pattern):

| Layer | Folder | Responsibility |
|-------|--------|----------------|
| **1️⃣ Routes Layer** | `app/routes/` | Defines API endpoints and HTTP methods |
| **2️⃣ Controller Layer** | `app/controllers/` | Handles request/response, calls services |
| **3️⃣ Service Layer** | `app/services/` | Contains business logic and feature engineering |
| **4️⃣ Repository Layer** | `app/repositories/` | Loads the trained model and ML artifacts |

**Schemas** (`app/schemas/`) are used for **Pydantic-based input validation and response formatting**.

```
Client Request
     ↓
[Routes] → [Controllers] → [Services] → [Repositories] → Trained Model
     ↑                                                          ↓
     └───────────────── JSON Response ──────────────────────────┘
```

---

## 🧰 Tech Stack

- **Python 3.10+**
- **FastAPI** — Web framework
- **Pydantic v2** — Data validation
- **scikit-learn** — ML model (Linear Regression, SimpleImputer)
- **pandas** — Data preprocessing
- **joblib** — Model serialization
- **kagglehub** — Dataset download
- **uvicorn** — ASGI server
- **pytest** — Testing framework

---

## 📁 Project Structure

```
house-price-ml-api/
│
├── app/
│   ├── main.py                         # App entrypoint
│   ├── __init__.py
│   │
│   ├── routes/                         # 1️⃣ ROUTES LAYER
│   │   ├── __init__.py
│   │   └── predict.py
│   │
│   ├── controllers/                    # 2️⃣ CONTROLLER LAYER
│   │   ├── __init__.py
│   │   └── prediction_controller.py
│   │
│   ├── services/                       # 3️⃣ SERVICE LAYER
│   │   ├── __init__.py
│   │   └── prediction_service.py
│   │
│   ├── repositories/                   # 4️⃣ REPOSITORY LAYER
│   │   ├── __init__.py
│   │   └── model_repository.py
│   │
│   └── schemas/                        # PYDANTIC SCHEMAS
│       ├── __init__.py
│       └── house.py
│
├── data/
│   └── house_data_clean.csv            # Cleaned dataset used for training
│
├── models/                             # ML ARTIFACTS
│   ├── house_columns.pkl
│   ├── house_imputer.pkl
│   └── house_model.pkl
│
├── scripts/
│   └── train.py                        # Training script
│
├── tests/
│   └── test_api.py                     # API tests
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **Source:** [Indian House Price Prediction Dataset](https://www.kaggle.com/datasets/srisyra02/house-price-prediction-dataset) on Kaggle
- **Size:** 250,000 rows × 23 columns
- **Target Variable:** `Price_in_Lakhs`
- **Key Features Used:**
  - `BHK`
  - `Size_in_SqFt`
  - `Price_per_SqFt`
  - `Year_Built`
  - `Parking_Space`
  - `City` (one-hot encoded → 42 city columns)

The dataset is downloaded automatically via `kagglehub` when `train.py` is executed.

---

## 🤖 Model Details

- **Algorithm:** Multiple Linear Regression
- **Preprocessing:** `SimpleImputer(strategy="mean")` for missing values
- **Features:** 47 total (5 base + 42 city one-hot columns)
- **Train/Test Split:** 80/20
- **Evaluation Metrics:**
  - **MAE:** 81.14
  - **RMSE:** 100.83
  - **R² Score:** 0.490

Model artifacts saved:
- `house_model.pkl` — Trained LinearRegression model
- `house_imputer.pkl` — Fitted SimpleImputer
- `house_columns.pkl` — Exact column order used during training

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Shlokverma0/house-price-api.git
cd house-price-api
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Kaggle credentials
To download the dataset, you need Kaggle API credentials:
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place `kaggle.json` in:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`

### 6. Train the model
```bash
python scripts/train.py
```
This will:
- Download the dataset from Kaggle
- Preprocess and clean the data
- Train the Linear Regression model
- Save `.pkl` artifacts to `models/` and `data/`

---

## 🚀 Running the API

Start the FastAPI server:
```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:
- **API Base URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Docs (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Root — API status check |
| `GET`  | `/health` | Health check (model loaded status) |
| `POST` | `/predict` | Predict house price |

### Request Body for `/predict`

| Field | Type | Description |
|-------|------|-------------|
| `BHK` | `int` | Number of Bedrooms, Hall, Kitchen |
| `Size_in_SqFt` | `float` | Property size in square feet |
| `Price_per_SqFt` | `float` | Price per square feet |
| `Year_Built` | `int` | Year property was built (1900–2026) |
| `Parking_Space` | `int` | 1 = Yes, 0 = No |
| `location` | `str` | City name (e.g., Mumbai, Delhi, Bangalore) |

---

## 📥 Example Request & Response

### Request (cURL)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "BHK": 3,
    "Size_in_SqFt": 1500,
    "Price_per_SqFt": 8000,
    "Year_Built": 2015,
    "Parking_Space": 1,
    "location": "Mumbai"
  }'
```

### Response
```json
{
  "predicted_price_in_lakhs": 154.28,
  "currency": "INR",
  "note": "Price is in Lakhs (1 Lakh = 100,000 INR)"
}
```

---

## 🧪 Running Tests

Run the API tests using `pytest`:
```bash
python -m pytest tests/test_api.py -v
```

Tests cover:
- Root endpoint (`/`)
- Health check endpoint (`/health`)
- Prediction endpoint (`/predict`)

---

## 👨‍💻 Author

**Shlok Verma**
- GitHub: [@Shlokverma0](https://github.com/Shlokverma0)
- LinkedIn: [shlok-verma](https://www.linkedin.com/in/shlok-verma-113713363)

---

## 📜 License

This project is created for educational purposes as part of a Machine Learning assignment.

---

## 🙏 Acknowledgements

- Kaggle for the [Indian House Price Prediction Dataset](https://www.kaggle.com/datasets/srisyra02/house-price-prediction-dataset)
- FastAPI for the excellent web framework
- scikit-learn for the ML tools
