import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib 
import os

def load_data():
    mercedes = pd.read_csv("data/raw/mercedes.csv")
    hyundai = pd.read_csv("data/raw/hyundai.csv")
    kia = pd.read_csv("data/raw/kia.csv")
    bmw = pd.read_csv("data/raw/bmw.csv")
    df = pd.concat([mercedes, hyundai, kia, bmw], ignore_index=True)
    df = shuffle(df, random_state=42)
    return df

def filter_top_models(df, top_n=6):
    """Top 6 Most Frequent Models by Make."""
    df_top_models = df.groupby("Make")["Model"].value_counts().groupby(level=0).head(top_n).reset_index()["Model"]
    return df[df["Model"].isin(df_top_models)]

def parse_engine(value):
    parts = value.split(" / ")
    engine_size = float(parts[0].replace(" L", "").strip()) if "L" in parts[0] else 0
    horsepower = int(parts[1 if "L" in parts[0] else 0].replace(" a.g.", "").strip())
    fuel_type = parts[2 if "L" in parts[0] else 1].strip()
    return pd.Series([engine_size, horsepower, fuel_type])

def clean_kilometer(value):
    return int(value.replace(" ", "").replace("km", ""))

def convert_to_azn(price):
    price = price.replace(" ", "")
    amount, currency = price[:-3], price[-3:]
    amount = int(amount)
    if currency == "USD":
        return amount * 1.7
    elif currency == "EUR":
        return amount * 1.9
    else:
        return amount

def apply_encoding_and_scaling(df):
    save_dir = os.path.join(os.path.dirname(__file__), "../../models", "preprocessing_objects")
    os.makedirs(save_dir, exist_ok=True)
    
    le = LabelEncoder()
    for col in ["Transmission", "Make", "New", "Fuel_Type", "Color"]:
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(save_dir, f"label_encoder_{col.lower()}.pkl"))

    df = pd.get_dummies(df, columns=["Model"], prefix="", prefix_sep="", dtype=int)

    numerical_columns = ['Kilometer', 'Engine_Size', 'Horsepower']
    scaler = RobustScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    joblib.dump(scaler, os.path.join(save_dir, "robust_scaler.pkl"))

    return df

def preprocess_data(df, apply_encoding=True):
    df = df.copy()
    df = df.drop_duplicates()

    # Feature extraction
    df[["Engine_Size", "Horsepower", "Fuel_Type"]] = df["Engine"].apply(parse_engine)
    df["Kilometer"] = df["Kilometer"].apply(clean_kilometer)
    df["Price"] = df["Price"].apply(convert_to_azn)
    df.drop(columns=["Engine"], inplace=True)

    if apply_encoding:
        df = apply_encoding_and_scaling(df)

    return df

def save_data(df, filename):
    """Saves the processed data as a CSV file."""
    df.to_csv(filename, index=False)

if __name__ == "__main__":

    df = load_data()

    df_filter = filter_top_models(df)
    save_data(df_filter, "data/interim/filtered_turbo_az.csv")

    df_clean = preprocess_data(df_filter, apply_encoding=False)
    save_data(df_clean, "data/interim/cleaned_turbo_az.csv")
    
    df_final = preprocess_data(df_filter, apply_encoding=True)
    save_data(df_final, "data/processed/prepared_turbo_az.csv")
    print("Data preprocessing completed and saved.")