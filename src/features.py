"""
Feature engineering for the Snitch Slow-Mover Prediction Model.

Derives STR_90, DOI, and the Slow_Mover target label from raw sales/stock
numbers, then drops those raw numeric columns before modeling so the
classifier only ever sees descriptive product ATTRIBUTES — the features
you'd know at the design/buy stage, before a SKU has any sales history.
This is what makes the model useful: it predicts slow-mover risk for a
planned SKU, not just labels SKUs that have already sold slowly.
"""
import pandas as pd
import numpy as np

ATTRIBUTE_FEATURES = [
    "Category", "Fabric", "Design", "Fit",
    "ASP_Bucket", "Sleeve_Type", "Size_Availability",
]

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="SKU_Data")

# Known typo/casing variants -> canonical value, from the injected real-world noise
FABRIC_CANON = {
    "polyester": "Polyester", "poly": "Polyester", "polyster": "Polyester",
    "cotton": "Cotton", "denim": "Denim", "linen": "Linen",
    "fleece": "Fleece", "blended": "Blended", "blend": "Blended",
}
CATEGORY_CANON = {
    "tshirt": "T-Shirts", "tshirts": "T-Shirts", "t shirts": "T-Shirts",
    "coords": "Co-ords", "co-ord": "Co-ords",
}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the real-world data-quality issues present in the raw export:
    inconsistent casing/typos, duplicate SKU rows, negative stock from
    returns-processing lag, and stale Size_Availability tags. See the
    Data_Quality_Notes sheet in the source workbook for the full list.
    """
    df = df.copy()

    # 1. Deduplicate on SKU_ID - keep the row with the higher (more complete) stock figure
    df = (
        df.sort_values("Sellable_Stock", ascending=False)
          .drop_duplicates(subset="SKU_ID", keep="first")
          .reset_index(drop=True)
    )

    # 2. Normalize casing/whitespace, then map known typos to canonical values
    df["Fabric"] = df["Fabric"].astype(str).str.strip()
    df["Fabric"] = df["Fabric"].apply(
        lambda v: FABRIC_CANON.get(v.lower(), v)
    )
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Category"] = df["Category"].apply(
        lambda v: CATEGORY_CANON.get(v.lower(), v)
    )

    # 3. Missing categorical values -> explicit "Unknown" bucket (not dropped)
    for col in ["Fit", "Sleeve_Type", "Size_Availability"]:
        df[col] = df[col].fillna("Unknown")

    # 4. Negative stock from returns-processing lag -> clip at 0
    df["Sellable_Stock"] = df["Sellable_Stock"].clip(lower=0)

    # 5. Outlier sales spikes (flash sales) -> cap at the 99th percentile so one
    #    spike doesn't distort STR_90/DOI for that SKU
    cap = df["L30_Sales_Qty"].quantile(0.99)
    df["L30_Sales_Qty"] = df["L30_Sales_Qty"].clip(upper=cap)

    return df

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)

    # --- Derived performance metrics (used ONLY to build the label) ---
    df["STR_90"] = np.where(
        df["Opening_Stock_90"] == 0, 0,
        df["Units_Sold_90"] / df["Opening_Stock_90"]
    )
    df["DOI"] = np.where(
        df["L30_Sales_Qty"] == 0, 999,
        df["Sellable_Stock"] / (df["L30_Sales_Qty"] / 30)
    )

    # --- Target label ---
    median_str90 = df["STR_90"].median()
    df["Slow_Mover"] = (
        (df["STR_90"] < median_str90) & (df["DOI"] > 120)
    ).astype(int)

    # --- Explicit interaction flags ---
    # Logistic regression is a linear (additive) model, so it can't discover
    # AND-combinations like "Printed AND Polyester" on its own from the two
    # main-effect columns. We engineer each researched risk pattern as its
    # own feature so the model can weight the *combination* directly, not
    # just each attribute independently. Each mirrors a documented apparel-
    # inventory risk pattern, not just one hand-picked combo:
    #   1. Printed + Polyester        - synthetic print combo, look fatigues fast (strongest)
    #   2. Fleece + premium ASP       - heavyweight fabric, short wear-window in a tropical market + higher price
    #   3. Linen + Shirts             - formal/niche aesthetic vs. a Gen-Z fast-fashion casual audience
    #   4. Oversized fit + outerwear  - fit trend that doesn't transfer to Jackets/Sweaters
    df["Is_Printed_Polyester"] = (
        (df["Design"] == "Printed") & (df["Fabric"] == "Polyester")
    ).astype(int)
    df["Is_Fleece_Premium"] = (
        (df["Fabric"] == "Fleece") & (df["ASP_Bucket"].isin(["1300-1599", "1600-1999"]))
    ).astype(int)
    df["Is_Linen_Shirt"] = (
        (df["Fabric"] == "Linen") & (df["Category"] == "Shirts")
    ).astype(int)
    df["Is_Oversized_Outerwear"] = (
        (df["Fit"] == "Oversized") & (df["Category"].isin(["Jackets", "Sweaters"]))
    ).astype(int)

    return df

INTERACTION_FEATURES = [
    "Is_Printed_Polyester", "Is_Fleece_Premium", "Is_Linen_Shirt", "Is_Oversized_Outerwear",
]

def get_model_frame(df: pd.DataFrame):
    """Return (X, y) using only attribute-level features (no leakage)."""
    features = ATTRIBUTE_FEATURES + INTERACTION_FEATURES
    X = df[features].copy()
    y = df["Slow_Mover"].copy()
    return X, y
