# scripts/deployment_gradio.py

import gradio as gr
import pandas as pd
import joblib  # Use joblib instead of pickle
import numpy as np
import os
import sys

class FraudPredictor:
    """
    Handles loading the fraud detection model and encoder,
    preprocessing input data, and making predictions.
    """
    def __init__(self, model_path, encoder_path):
        """
        Initializes the predictor by loading the model and encoder.
        """
        try:
            # Load model using joblib instead of pickle
            self.model = joblib.load(model_path)
            print("Model loaded successfully.")
            
            # Load encoder using pickle (as in your preprocessing script)
            import pickle
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
            print("Encoder loaded successfully.")
            
            # Get payment methods from encoder
            if hasattr(self.encoder, 'categories_'):
                self.payment_methods = list(self.encoder.categories_[0])
                print(f"Available payment methods: {self.payment_methods}")
            else:
                # Default payment methods if not available
                self.payment_methods = ["Credit Card", "PayPal", "Debit Card"]
                print(f"Using default payment methods: {self.payment_methods}")
            
            # Get feature names from model if available
            if hasattr(self.model, 'feature_names_in_'):
                self.model_feature_names = self.model.feature_names_in_
                print(f"Using model's feature names: {self.model_feature_names}")
            else:
                # Get column names from the training data
                # This is based on your preprocessing script
                numeric_features = ['accountAgeDays', 'numItems', 'localTime', 'paymentMethodAgeDays']
                
                # Since you used drop='first' in your encoder, we don't include the first category
                encoded_features = [f'paymentMethod_{method}' for method in self.payment_methods[1:]]
                
                self.model_feature_names = numeric_features + encoded_features
                print(f"Using manually defined feature names: {self.model_feature_names}")
                
        except Exception as e:
            print(f"Error initializing predictor: {e}", file=sys.stderr)
            raise

    def predict(self, account_age, num_items, local_time, payment_method, payment_method_age):
        """
        Processes inputs and makes a prediction.
        """
        # Create input dataframe
        input_df = pd.DataFrame({
            "accountAgeDays": [float(account_age)],
            "numItems": [int(num_items)],
            "localTime": [float(local_time)],
            "paymentMethod": [str(payment_method)],
            "paymentMethodAgeDays": [float(payment_method_age)]
        })
        
        try:
            # Process input
            processed_input = self.transform_input(input_df)
            print(f"Processed input shape: {processed_input.shape}")
            print(f"Processed input columns: {processed_input.columns.tolist()}")
            
            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            probability = self.model.predict_proba(processed_input)[0][1]
            
            # Format result
            result = f"{'Fraudulent' if prediction == 1 else 'Not Fraudulent'} (Probability: {probability:.2f})"
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}", file=sys.stderr)
            print(f"Input data: {input_df}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def transform_input(self, input_df):
        """
        Transforms the input data for the model.
        """
        # Extract features
        numeric_features = input_df[['accountAgeDays', 'numItems', 'localTime', 'paymentMethodAgeDays']]
        categorical_features = input_df[['paymentMethod']]
        
        # Encode categorical features
        encoded_features = self.encoder.transform(categorical_features)
        
        # Get encoded column names
        encoded_columns = self.encoder.get_feature_names_out(['paymentMethod'])
        
        # Convert to dense array if sparse
        if hasattr(encoded_features, 'toarray'):
            encoded_features = encoded_features.toarray()
            
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoded_columns,
            index=input_df.index
        )
        
        # Combine numeric and encoded features
        final_df = pd.concat([numeric_features, encoded_df], axis=1)
        
        # Ensure all expected columns are present
        for col in self.model_feature_names:
            if col not in final_df.columns:
                final_df[col] = 0
        
        # Remove any extra columns not expected by the model
        extra_cols = [col for col in final_df.columns if col not in self.model_feature_names]
        if extra_cols:
            print(f"Removing extra columns not used during training: {extra_cols}")
            final_df = final_df.drop(columns=extra_cols)
        
        # Reorder columns to match model expectations
        final_df = final_df[self.model_feature_names]
        
        return final_df

# Define paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Initialize predictor
try:
    predictor = FraudPredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
except Exception as e:
    print(f"Failed to initialize FraudPredictor: {e}", file=sys.stderr)
    sys.exit(1)

# Define Gradio interface
inputs = [
    gr.Number(label="Account Age (Days)", value=100),
    gr.Number(label="Number of Items", value=2),
    gr.Number(label="Local Time", value=14.5),
    gr.Dropdown(
        choices=predictor.payment_methods,
        label="Payment Method",
        value=predictor.payment_methods[0] if predictor.payment_methods else "Credit Card"
    ),
    gr.Number(label="Payment Method Age (Days)", value=50)
]

outputs = gr.Textbox(label="Prediction Result")

# Create interface
interface = gr.Interface(
    fn=predictor.predict,
    inputs=inputs,
    outputs=outputs,
    title="Fraud Detection Predictor",
    description="Enter transaction details to check if it's potentially fraudulent."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()