"""
Builds and trains the Slow-Mover logistic regression pipeline.
Trained fresh on app startup (and cached) rather than shipping a pickled
model file, so there's no scikit-learn version-mismatch risk between where
the model was trained and where the app is hosted.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.features import ATTRIBUTE_FEATURES, INTERACTION_FEATURES


def build_and_train(df: pd.DataFrame):
    """
    Trains the pipeline on the full engineered dataframe and returns
    (pipeline, metrics_dict, feature_importance_df).
    """
    features = ATTRIBUTE_FEATURES + INTERACTION_FEATURES
    X = df[features].copy()
    y = df["Slow_Mover"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ATTRIBUTE_FEATURES)],
        remainder="passthrough",
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Test SKUs": len(y_test),
        "Overall slow-mover rate": y.mean(),
    }

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    return pipeline, metrics, importance


def explain_prediction(pipeline, input_row: pd.DataFrame, top_n: int = 3):
    """
    Returns the top contributing features (by coefficient * activation) for a
    single-row prediction, as a list of (feature_name, contribution) tuples.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["model"]

    transformed = preprocessor.transform(input_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)[0]

    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]
    contributions = transformed * coefs

    order = np.argsort(contributions)[::-1]
    top_positive = [
        (feature_names[i], contributions[i])
        for i in order if contributions[i] > 0
    ][:top_n]
    return top_positive


FRIENDLY_NAMES = {
    "Is_Printed_Polyester": "Printed design on Polyester fabric",
    "Is_Fleece_Premium": "Fleece fabric priced at ₹1,300+",
    "Is_Linen_Shirt": "Linen fabric in the Shirts category",
    "Is_Oversized_Outerwear": "Oversized fit in Jackets/Sweaters",
}


def friendly_feature_name(raw_name: str) -> str:
    """Converts a pipeline feature name like 'cat__Fabric_Polyester' or
    'remainder__Is_Printed_Polyester' into a planner-readable label."""
    name = raw_name.split("__", 1)[-1]
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]
    if "_" in name:
        col, val = name.split("_", 1)
        return f"{col.replace('_', ' ')} = {val}"
    return name
