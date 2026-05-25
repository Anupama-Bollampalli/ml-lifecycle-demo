# ML Lifecycle Demo

Full ML lifecycle: data exploration → feature engineering → train 3 models → compare → deploy as prediction API → interactive frontend.

Uses the UCI Breast Cancer Wisconsin dataset (built-in with scikit-learn).

## Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble, feature importance |
| Neural Network | PyTorch, 30→64→32→1 |

## Architecture

```
ml-lifecycle-demo/
├── backend/
│   ├── feature_engineering.py   # Data loading, scaling
│   ├── models/
│   │   ├── logistic_model.py    # sklearn LogisticRegression wrapper
│   │   ├── forest_model.py      # sklearn RandomForest wrapper
│   │   └── neural_model.py      # PyTorch neural network
│   ├── evaluate.py              # Metrics computation
│   ├── train.py                 # Training pipeline
│   ├── main.py                  # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 5-tab layout
│   │   └── components/
│   │       ├── ModelComparison.tsx
│   │       ├── PredictionPanel.tsx
│   │       ├── ConfusionMatrix.tsx
│   │       ├── FeatureImportance.tsx
│   │       └── LearningCurve.tsx
│   └── package.json
└── saved_models/                # Generated after training
```

## Setup & Run

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# API runs at http://localhost:8003
```

Training runs automatically on first startup. To retrain manually:
```bash
cd backend
python train.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173/ml-lifecycle-demo/
```

### Docker

```bash
docker build -t ml-lifecycle-demo .
docker run -p 7863:7863 ml-lifecycle-demo
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /models/comparison` | All 3 models' metrics |
| `GET /models/{model}/confusion-matrix` | Confusion matrix for a model |
| `GET /models/{model}/learning-curve` | Learning curve data |
| `GET /features/importance` | Top 15 feature importances |
| `POST /predict` | Predict on a feature vector |

## Tech Stack

- **Backend**: FastAPI, scikit-learn, PyTorch, NumPy, joblib
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Recharts
- **Dataset**: UCI Breast Cancer Wisconsin (569 samples, 30 features)

## Live Demo

- **Frontend**: [GitHub Pages](https://anupama-bollampalli.github.io/ml-lifecycle-demo/)
- **Backend API**: [HF Space](https://abollampalli-ml-lifecycle-demo.hf.space)
