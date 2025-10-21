from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from utils.readcvs import X, y, data
from utils.gradient_descent import gradient_descent_fit

# -------------------------------
# Train model once at startup
# -------------------------------
B_final = gradient_descent_fit(X, y)
print("Model training completed!")
print(f"Final weights: {B_final}")

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Store Sales Predictor API")

# CORS middleware - ADD YOUR FRONTEND PORT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5174", "http://127.0.0.1:3000", "http://127.0.0.1:5174",
                   "https://store-sales.netlify.app",
        "https://store-sales-api-1.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Pydantic models
# -------------------------------
class SalesInput(BaseModel):
    TotalGasGallons: float
    LottoSales: float
    DayType: int  # 0 = weekday, 1 = weekend

class SalesBatchInput(BaseModel):
    data: List[SalesInput]

class ModelInfoResponse(BaseModel):
    weights: List[float]
    features: List[str]
    data_points: int

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Store Sales Predictor API"}

# -------------------------------
# Single prediction
# -------------------------------
@app.post("/predict")
def predict_sales(sales: SalesInput):
    X_new = [1, sales.TotalGasGallons, sales.LottoSales, sales.DayType]
    prediction = float(np.dot(X_new, B_final))
    return {
        "predicted_sales": round(prediction, 2),
        "input_features": {
            "TotalGasGallons": sales.TotalGasGallons,
            "LottoSales": sales.LottoSales,
            "DayType": "Weekend" if sales.DayType == 1 else "Weekday"
        }
    }

# -------------------------------
# Batch predictions for charting
# -------------------------------
@app.post("/predict_batch")
def predict_batch(sales_batch: SalesBatchInput):
    predictions = []
    for s in sales_batch.data:
        X_new = [1, s.TotalGasGallons, s.LottoSales, s.DayType]
        predictions.append(float(np.dot(X_new, B_final)))
    return {"predicted_sales": predictions}

# -------------------------------
# Historical data
# -------------------------------
@app.get("/history")
def get_history():
    history = data.to_dict(orient="records")
    return {"history": history, "total_records": len(history)}

# -------------------------------
# Model information
# -------------------------------
@app.get("/model_info")
def get_model_info():
    return ModelInfoResponse(
        weights=B_final.tolist(),
        features=["Intercept", "TotalGasGallons", "LottoSales", "DayType"],
        data_points=len(data)
    )

# -------------------------------
# Sample prediction ranges for frontend sliders
# -------------------------------
@app.get("/sample_ranges")
def get_sample_ranges():
    return {
        "TotalGasGallons": {
            "min": float(data['TotalGasGallons'].min()),
            "max": float(data['TotalGasGallons'].max()),
            "mean": float(data['TotalGasGallons'].mean())
        },
        "LottoSales": {
            "min": float(data['LottoSales'].min()),
            "max": float(data['LottoSales'].max()),
            "mean": float(data['LottoSales'].mean())
        }
    }

# -------------------------------
# Weekly sales JSON for React chart
# -------------------------------
@app.get("/weekly_sales_json")
def weekly_sales_json():
    weeks = list(range(1, len(data) + 1))  # simple week numbers
    total_sales = data["TotalSales"].tolist()
    gas_sales = data["TotalGasGallons"].tolist()
    lotto_sales = data["LottoSales"].tolist()
    predicted_sales = [
        float(np.dot([1, row["TotalGasGallons"], row["LottoSales"], row["DayType"]], B_final))
        for _, row in data.iterrows()
    ]

    return {
        "weeks": weeks,
        "total_sales": total_sales,
        "gas_sales": gas_sales,
        "lotto_sales": lotto_sales,
        "predicted_sales": predicted_sales
    }

# -------------------------------
# ADD THIS ENDPOINT - Chart data that frontend expects
# -------------------------------
@app.get("/chart_data")
def get_chart_data():
    """Endpoint that matches what the frontend expects"""
    chart_data = []
    for i, (_, row) in enumerate(data.iterrows()):
        chart_data.append({
            "name": f"Day {i+1}",
            "TotalSales": float(row["TotalSales"]),
            "GasSold": float(row["TotalGasGallons"]),
            "Lotto": float(row["LottoSales"]),
            "DayType": "Weekend" if row["DayType"] == 1 else "Weekday"
        })
    return chart_data

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
