# Store Sales Predictor API 🏪

FastAPI backend for gas station sales predictions using custom gradient descent.

## 🚀 Live URLs
- **Frontend**: https://store-sales.netlify.app
- **Backend**: https://store-sales-api-1.onrender.com
- **API Docs**: https://store-sales-api-1.onrender.com/docs

## 📦 Features
- ML sales predictions (gas gallons, lotto sales, day type)
- REST API with auto-generated docs
- CORS enabled for frontend
- Production deployment on Render

## 🔧 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000
```

## 📋 Requirements
```
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
pandas==2.0.3
```

## 🔌 Main Endpoints
- `POST /predict` - Get sales prediction
- `GET /history` - Historical data  
- `GET /model_info` - Model weights
- `GET /health` - API status

## 🏗️ Project Structure
```
main.py                 # FastAPI app
gradient_descent.py     # Custom ML algorithm  
readcvs.py             # Data loading
requirements.txt       # Dependencies
```

## 🚀 Deployment
Deployed on Render with automatic deployments from GitHub.

---

**Built with FastAPI & Custom Gradient Descent**
