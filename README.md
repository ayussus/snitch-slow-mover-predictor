# Snitch Slow-Mover Prediction Model

Predicts slow-mover risk for a planned SKU **before it has any sales history** —
using only design attributes (fabric, design, fit, category, price bucket,
sleeve type, size availability). Built for Snitch's merchandising/planning
team to catch overstock risk at the buy stage instead of 60–90 days after
launch.

**Live app:** _add your Streamlit Cloud URL here after deploying_

## What's in this repo

```
snitch-slow-mover-predictor/
├── app.py                  # Streamlit app (3 tabs: findings, single-SKU scorer, bulk scorer)
├── requirements.txt
├── data/
│   └── snitch_slow_mover_dataset.xlsx   # synthetic training dataset + README/Data_Quality_Notes sheets
├── src/
│   ├── features.py          # data cleaning, feature engineering, target label derivation
│   └── model.py              # pipeline builder, training, feature-importance explanation
└── README.md
```

## How the model works

1. **Target label** is derived (not stored in the raw data): a SKU is a
   `Slow_Mover` if `STR_90 < median(STR_90)` AND `DOI > 120`.
2. **Model features are attributes only** — Category, Fabric, Design, Fit,
   ASP_Bucket, Sleeve_Type, Size_Availability — never the raw sales/stock
   numbers, since those directly define the label (using them would leak
   the answer into the input).
3. Four attribute combinations are the strongest learned risk signals:
   - Printed + Polyester
   - Fleece fabric + ₹1,300+ price
   - Linen + Shirts
   - Oversized fit + Jackets/Sweaters
4. Model: **Logistic Regression** (`class_weight="balanced"`, one-hot
   encoded categoricals). Chosen over a shallow XGBoost comparison because
   accuracy was comparable and logistic regression's coefficients are
   directly explainable to a non-technical planning team ("Fleece priced
   above ₹1,300 is ~30x more likely to be a slow-mover").

The model is **trained fresh each time the app starts** (and cached with
`st.cache_resource`) rather than shipped as a pickled file — this avoids
scikit-learn version-mismatch issues between your training environment and
Streamlit Cloud's, and keeps the repo free of binary model files.

## Running locally

```bash
git clone https://github.com/<your-username>/snitch-slow-mover-predictor.git
cd snitch-slow-mover-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploying for free (Streamlit Community Cloud)

No paid plan needed — Streamlit Community Cloud is free for public GitHub repos.

1. Push this repo to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select this repo → branch `main` → main file path `app.py`.
4. Click **Deploy**. Build takes ~2–3 minutes.
5. Every push to `main` auto-redeploys the app.

## Using the app

- **Key Findings tab** — model accuracy metrics and the 4 risk patterns, in plain language.
- **Score a Single SKU tab** — pick attributes for one planned SKU, get a risk score + reasons + suggested action.
- **Bulk Portfolio Scoring tab** — upload an Excel of an upcoming collection's planned SKUs (template provided), get every SKU scored, a risk breakdown, and a downloadable results file.

## Data note

`data/snitch_slow_mover_dataset.xlsx` is **synthetic data** engineered to
mirror realistic Snitch SKU-level patterns (see the `README` and
`Data_Quality_Notes` sheets inside the workbook for the full data dictionary,
the deliberate risk signals, and the real-world data-quality issues modeled
in). It is not live Snitch sales data.
