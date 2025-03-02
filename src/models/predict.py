import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    """Load the trained model from a file."""
    return joblib.load(model_path)

def preprocess_input_data(input_data):
    """Preprocess the input data to match the training data format."""
    saved_models_object = os.path.join(os.path.dirname(__file__), "../../models", "preprocessing_objects")

    label_encoders = {
        "Make": "label_encoder_make.pkl",
        "Color": "label_encoder_color.pkl",
        "Transmission": "label_encoder_transmission.pkl",
        "Fuel_Type": "label_encoder_fuel_type.pkl",
        "New": "label_encoder_new.pkl"
    }

    for column, encoder_file in label_encoders.items():
        encoder = joblib.load(os.path.join(saved_models_object, encoder_file))
        input_data[column] = encoder.transform([input_data[column]])[0]

    scaler = joblib.load(os.path.join(saved_models_object, "robust_scaler.pkl"))
    scaled_values = scaler.transform([[input_data['Kilometer'], input_data['Engine_Size'], input_data['Horsepower']]])
    input_data['Kilometer'], input_data['Engine_Size'], input_data['Horsepower'] = scaled_values[0]  

    model_columns = [
        '330', '520', '530',
        '530e', 'Accent', 'Carnival', 'Cerato', 'E 220 d', 'E 300 de',
        'E 300 e', 'Elantra', 'Forte', 'G 63 AMG', 'GLE 350 4MATIC',
        'GLS 450 4MATIC', 'Grandeur', 'K5', 'Santa Fe', 'Sonata',
        'Sorento', 'Sportage', 'Tucson', 'X5', 'X7']
    
   
    model_column = f"{input_data['Model']}"
    if model_column not in model_columns:
        raise ValueError(f"Invalid car model: {model_column}")

    # Fill all model columns with 0
    model_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    model_data[model_column] = 1
    # Merge with other features
    input_data = pd.concat([pd.DataFrame([input_data]).drop(columns=['Model']), model_data], axis=1)

    # Use the column names that the model saw during training
    expected_columns = [
        'Make', 'Year', 'Color', 'Kilometer', 'Transmission', 'New',
       'Engine_Size', 'Horsepower', 'Fuel_Type', '330', '520', '530',
       '530e', 'Accent', 'Carnival', 'Cerato', 'E 220 d', 'E 300 de',
       'E 300 e', 'Elantra', 'Forte', 'G 63 AMG', 'GLE 350 4MATIC',
       'GLS 450 4MATIC', 'Grandeur', 'K5', 'Santa Fe', 'Sonata',
       'Sorento', 'Sportage', 'Tucson', 'X5', 'X7']

    # Add missing columns (if not already present)
    for column in expected_columns:
        if column not in input_data.columns:
            input_data[column] = 0

    input_data = input_data[expected_columns]
    return input_data

def predict_price(model, input_data):
    """Use the loaded model to predict price for new data."""
    return model.predict(input_data)

def main():
    # Sample input data
    sample_data = {
        'Make': 'Hyundai',
        'Model': 'Sonata',
        'Year': 2021,
        'Color': 'Yaş Asfalt',
        'Kilometer': 53000,
        'Transmission': 'Avtomat',
        'New': 'Xeyr',
        'Engine_Size': 2.0,
        'Horsepower': 192,
        'Fuel_Type': 'Hibrid'
    }

    # Preprocess the input data
    input_data = preprocess_input_data(sample_data)

    current_dir = os.path.join(os.path.dirname(__file__), "../../models",  "trained")
    model_path = os.path.join(current_dir, "stacking_regressor.pkl")

    model = load_model(model_path)

    prediction = predict_price(model, input_data)
    print("Predicted Price:", prediction[0].round(), "AZN")

if __name__ == "__main__":
    main()
