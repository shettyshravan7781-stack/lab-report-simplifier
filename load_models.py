import joblib  # or import pickle

# 1. Path to one of your saved PKL models
model_path = r"C:\AI_Lab_Report\output\saved_models\xgboost_Target_Anemia_Risk.pkl"

# 2. Load the model
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print("Model object type:", type(model))

    # If it's a trained XGBoost model object, print basic properties
    if hasattr(model, "n_features_in_"):
        print("Expected input features count:", model.n_features_in_)

    # Get feature names if saved inside
    if hasattr(model, "feature_names_in_"):
        print("\nFeature Names:")
        print(model.feature_names_in_)

except Exception as e:
    print("Error loading model:", e)