from pathlib import Path
import joblib
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "dataset" / "online_retail_II.csv"
ARTIFACTS = BASE_DIR / "models" / "artifacts"

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV



def load()-> pd.DataFrame:   
    df = pd.read_csv(DEFAULT_CSV)
    return clean_df(df)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Quantity'] > 0]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype('int64')
    df['InvoiceDate'] = df["InvoiceDate"].astype('datetime64[ns]')
    df['Invoice'] = df['Invoice'].astype(str)
    df = df[~df['Invoice'].str.startswith('C', na=False)]
    drop_non_numeric = ['ADJUST','ADJUST2','BANK CHARGES','C2','DOT','M','PADS','POST','TEST001','TEST002']
    df = df[~df['StockCode'].isin(drop_non_numeric)]
    df = df.drop_duplicates()
    df = df[df["Price"] > 0]
    df["TotalPrice"] = df['Price'] * df['Quantity']
    return df

def build_survival_df(df : pd.DataFrame) -> pd.DataFrame:
    survival = df.groupby('Customer ID').agg(
        first_purchase_date = ('InvoiceDate', 'min'),
        last_purchase_date = ('InvoiceDate', 'max'),
        TotalOrders = ('Invoice', 'nunique'),
        TotalSpent = ('TotalPrice', 'sum'),
        country = ('Country', 'first')).reset_index()
    survivalend = df['InvoiceDate'].max()
    survival['duration'] = (survival['last_purchase_date'] - survival['first_purchase_date']).dt.days
    
    survival['event'] = survival.apply(lambda x : 1 if (survivalend - x['last_purchase_date']).days > 90 else 0, axis=1)
    survival['is_uk'] = survival['country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)
    return survival

def build_aft_df(survival : pd.DataFrame) -> pd.DataFrame:
    aft_df = survival.select_dtypes(include='number')
    aft_df.drop(columns=['Customer ID'], inplace=True)
    aft_df['logTotalOrders'] = np.log1p(aft_df['TotalOrders'])
    aft_df.drop(columns=['TotalSpent', 'TotalOrders'],inplace=True)
    aft_df['duration'] = aft_df['duration'].where(aft_df['duration'] > 0, 0.5)
    return aft_df

def  train_weibull(aft_df : pd.DataFrame)-> WeibullAFTFitter:
    AFT = WeibullAFTFitter()
    AFT.fit(aft_df, duration_col='duration', event_col='event')
    return (AFT)

def build_rfm_df(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df['InvoiceDate'].max() - pd.Timedelta(days = 180)
    churn_date = snapshot_date + pd.Timedelta(days = 90)
    feature_window = df[df['InvoiceDate'] <= snapshot_date]
    active = df[df['InvoiceDate']>snapshot_date]['Customer ID'].unique()
    
    rfm_df = feature_window.groupby('Customer ID').agg(
        recency = ('InvoiceDate',lambda x : (snapshot_date - x.max()).days),
        frequency = ('Invoice', 'nunique'),
        monetary = ('TotalPrice', 'sum'),
        avg_order_value = ('TotalPrice', 'mean'),
        avg_quantity = ('Quantity', 'mean'),
        unique_products = ('StockCode', 'nunique'),
        top_category=('StockCode', lambda x: x.mode()[0])
    ).reset_index()
    
    rfm_df['event'] = rfm_df['Customer ID'].apply(lambda x : 0 if x in active else 1)
    return rfm_df

def train_rf(rfm_df: pd.DataFrame)->RandomizedSearchCV:
    
    X = rfm_df.drop(columns=['Customer ID', 'event','top_category'])
    Y = rfm_df['event']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    rfc = RandomizedSearchCV(RandomForestClassifier(), param_dist,
                            n_iter=20, cv=5, random_state=42, n_jobs=-1)
    rfc.fit(X_train,Y_train)
    return rfc

if __name__ == "__main__":
    df = load()
    survival_df = build_survival_df(df)
    rfm_df = build_rfm_df(df)
    aft_df = build_aft_df(survival_df)
    aft_model = train_weibull(aft_df)
    rfc = train_rf(rfm_df)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(aft_model, ARTIFACTS / "weibull_aft.joblib")
    joblib.dump(rfc, ARTIFACTS / "rf_model.joblib")