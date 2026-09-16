# 👕 Snitch Slow-Mover Prediction Model

**Live app:** https://snitch-slow-mover-predictor.streamlit.app/

A tool that predicts whether a **planned SKU will become a slow-mover — before it's even manufactured or stocked.** Built for Snitch's merchandising/planning team to catch overstock risk at the buying stage instead of finding out 60–90 days after launch.

---

## The problem this solves

Normally, a brand only learns a SKU is a "slow-mover" (low sales, high stock) months after it's already been bought, printed, and shipped to warehouses — by which point the money is already spent and markdowns are the only option left.

This model flips that around: give it a **planned SKU's attributes** (fabric, fit, design, category, price point, etc.) — attributes you know *before* placing a manufacturing order — and it tells you how risky that SKU profile has historically been, based on patterns learned from past SKU performance.

---

## How it works, in plain terms

1. **We start with SKU-level data**: for every past SKU, we know its attributes (Category, Fabric, Design, Fit, price bucket, etc.) and its sales/stock numbers (units sold, stock on hand).

2. **We compute two performance metrics for every SKU:**
   - **STR-90** (Sell-Through Rate, 90 days) — what % of stock actually sold
   - **DOI** (Days of Inventory) — how many days the current stock would last at the recent sales pace

3. **A SKU is labeled "Slow-Mover" if:**
   `STR-90 is below the median` **AND** `DOI is above 120 days`
   (i.e., it sold worse than average AND is carrying a large stock overhang)

4. **The model is trained to predict this label using ONLY the attributes** — never the sales/stock numbers themselves. This is on purpose: if the model saw sales numbers, it would just be reading the answer off the label instead of learning anything useful. Attributes are the only thing you actually know *before* a SKU launches, so that's the only thing the model is allowed to use.

5. **Output:** for any new/planned SKU, the model gives a probability (0–100%) that it will become a slow-mover, plus the specific attribute combination driving that risk.

---

## The 4 risk patterns the model learned

These aren't random correlations — each mirrors a real, explainable apparel-inventory risk cause, and the dataset was deliberately built with these patterns so the model has real signal to learn from:

| Pattern | Why it's risky | Learned risk multiplier |
|---|---|---|
| **Printed + Polyester** | Cheap synthetic print combo — the look fatigues fast in a fast-fashion cycle | ~99x more likely to be a slow-mover |
| **Fleece + ₹1,300+ price** | Heavyweight fabric has a short wear-window in a mostly-tropical market; a premium price on top narrows the buyer pool further | ~30x |
| **Linen + Shirts** | Formal/niche aesthetic that doesn't fit a Gen-Z fast-fashion casual audience | ~23x |
| **Oversized fit + Jackets/Sweaters** | Oversized fits work in tees/hoodies but not outerwear, where fit misjudgment leaves fringe-size overstock | ~12x |

The model also picks up **weaker, unofficial signals** for combinations it was never explicitly told about — e.g. Fleece + Shorts (a thermally mismatched pairing) scores meaningfully above baseline even without a dedicated rule for it, just from combining what it separately learned about each attribute. This is a good sign: it means the model generalizes a little beyond exactly what it was shown, though not as confidently as the 4 patterns it was explicitly trained on.

---

## Why the data isn't perfectly clean (and why that's intentional)

Real retail data is messy — incomplete tags, inconsistent spelling across teams, sync lags, one-off sales spikes. The training dataset deliberately includes ~15-20% of rows with realistic data-quality issues (see the `Data_Quality_Notes` sheet inside the Excel file for the full list and rates), and the pipeline cleans them the way a real system would: standardizing spelling, imputing missing tags as "Unknown," deduplicating synced records, clipping impossible negative stock, and capping one-off sales spikes. This makes the model more robust to the kind of messiness it'll actually see on live data, rather than only ever working on a pristine sample.

---

## Why Logistic Regression (not XGBoost)

Both were tested side-by-side. Logistic Regression was chosen for the deployed app because:
- Accuracy was comparable to a shallow XGBoost model (XGBoost edges ahead by a few points on clean data, roughly tied on messy/realistic data)
- Its output is **directly explainable to a non-technical planning team** — "Fleece priced above ₹1,300 is ~30x more likely to be a slow-mover" is a sentence someone can act on, versus a tree-model importance score that needs extra tooling (like SHAP) to explain properly

---

## Repo structure

```
snitch-slow-mover-predictor/
├── app.py                  # Streamlit app (3 tabs: findings, single-SKU scorer, bulk scorer)
├── requirements.txt
├── data/
│   └── snitch_slow_mover_dataset.xlsx   # training dataset + README/Data_Quality_Notes sheets
├── src/
│   ├── features.py          # data cleaning, feature engineering, target label derivation
│   └── model.py              # pipeline builder, training, feature-importance explanation
└── README.md
```

The model is **trained fresh every time the app starts** (and cached) rather than shipped as a pickled file — avoids version-mismatch issues between environments and keeps the repo free of binary model files.

---

## Using the app

Open **https://snitch-slow-mover-predictor.streamlit.app/** — no login or setup needed.

- **🏠 Key Findings** — model accuracy at a glance, the 4 risk patterns explained, and a category-level risk chart
- **🔍 Score a Single SKU** — pick attributes for one planned SKU → get a risk %, the specific reasons, and a suggested buying action
- **📦 Bulk Portfolio Scoring** — upload an Excel of an upcoming collection's planned SKUs (template provided in-app) → every SKU scored, risk breakdown by category, downloadable results

Required columns for bulk upload: `Category, Fabric, Design, Fit, ASP_Bucket, Sleeve_Type, Size_Availability` (one row per planned SKU — no sales data needed, since these are pre-launch SKUs).

---

## Limitations to keep in mind

- **Synthetic data**: trained on data engineered to mirror realistic Snitch-style patterns, not live sales data. Once real historical data is available, retrain on that for production use.
- **Only 4 explicit risk rules**: the model catches other risky combinations only weakly (through general attribute correlation, not a dedicated rule). If a new pattern shows up consistently in real data, it should be added as an explicit interaction flag (same pattern as the existing 4 in `src/features.py`) and the model retrained.
- **Not a sales forecast**: this predicts *risk category*, not exact units sold or exact timing.
