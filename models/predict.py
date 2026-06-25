from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .train import load, clean_df, build_survival_df,build_aft_df, build_rfm_df

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = BASE_DIR / "models" /"artifacts"


def predict_new_data(df : pd.DataFrame) -> pd.DataFrame : 
    df = clean_df(df)
    rf, aft = joblib.load(ARTIFACTS / "rf_model.joblib"), joblib.load(ARTIFACTS / "weibull_aft.joblib")
    rfm_df = build_rfm_df(df)
    X_rf = rfm_df[['recency','frequency','monetary','avg_order_value','avg_quantity','unique_products']]
    rfm_df['churn_probability'] = rf.predict_proba(X_rf)[:, 1]
    survival_df = build_survival_df(df)
    aft_df = build_aft_df(survival_df)
    X_aft = aft_df[['logTotalOrders', 'is_uk']]
    survival_df['expected_lifetime_days'] = aft.predict_median(X_aft).values
    result = rfm_df[['Customer ID', 'churn_probability']].merge(
    survival_df[['Customer ID', 'expected_lifetime_days']], on='Customer ID'
    )
    return result

def lookup_customer(customer_id: int):
    df = load()
    customer_df = df[df['Customer ID'] == customer_id]
    if customer_df.empty:
        return {"error": f"Customer {customer_id} not found"}
    return customer_df.to_dict(orient='records')

if __name__ == "__main__":
    # Test predict_new_data
    df = pd.read_csv(BASE_DIR / "dataset" / "synthetic_retail.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    result = predict_new_data(df)
    print(result.head())

    # Test lookup_customer
    info = lookup_customer(12347)
    print(info)
    