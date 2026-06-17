# LoanIQ — ML Loan Approval Predictor

A loan approval predictor with two independent parts:

1. **Frontend (`public/index.html`)** — fully self-contained. The trained
   Random Forest (scaler + all 100 trees) is embedded directly as JSON in the
   page, and predictions run as pure JavaScript in the browser. No backend
   call required — this is why the page works offline and on GitHub Pages.
2. **Backend (`app.py`)** — a real FastAPI prediction API (`/predict`,
   `/health`) backed by `model.pkl`, available if you want to call
   predictions from another client (mobile app, other site, etc.). The
   current frontend does not call this — it's optional infrastructure.

## Project Structure
```
osuna/
├── app.py              # FastAPI backend (Vercel entrypoint)
├── train_model.py      # Trains the model & writes model.pkl
├── model.pkl           # Trained model
├── loan_dataset.csv    # Training dataset
├── requirements.txt    # Python dependencies
├── public/
│   └── index.html      # Self-contained frontend (served statically)
└── README.md
```

## Deploying to Vercel

Vercel auto-detects a FastAPI app when it finds `fastapi` in
`requirements.txt` and a `FastAPI` instance named `app` at a supported
entrypoint — `app.py` at the repo root qualifies, so **no `vercel.json` is
needed**.

1. Push this repo to GitHub.
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import the
   repo.
3. Click **Deploy**. That's it.

Everything in `public/` (your frontend) is served directly from Vercel's CDN
at the site root. `app.py` becomes a single Vercel Function handling
`/predict` and `/health`.

### Local development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
vercel dev
```

## Retraining the model
```bash
pip install -r requirements.txt
python train_model.py   # regenerates loan_dataset.csv and model.pkl
```
If you retrain, also regenerate the embedded `MODEL_DATA` JSON in
`public/index.html` to match, since the frontend doesn't read `model.pkl`
directly.

## API Endpoints
| Method | URL        | Description          |
|--------|------------|-----------------------|
| GET    | /health    | Health check          |
| POST   | /predict   | Get loan prediction   |

## Sample API Call
```bash
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "income": 60000,
    "credit_score": 720,
    "loan_amount": 15000,
    "employment_years": 5,
    "num_accounts": 3,
    "num_loans": 1,
    "monthly_expenses": 1500,
    "savings": 20000,
    "debt_ratio": 0.3
  }'
```

## Model Info
- **Algorithm:** Random Forest Classifier (100 trees)
- **Dataset:** 1000 synthetic loan applications, 10 features
- **Accuracy:** ~96.5%
- **Features:** age, income, credit score, loan amount, employment years, accounts, loans, expenses, savings, debt ratio
