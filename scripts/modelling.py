import numpy as np
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


df = pd.read_csv("data/preprocessed_data.csv")

X = df.drop("label", axis = 1)
y = df["label"]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Handing class imbalance through Random Forest - class_weight parameter

class_weights = compute_class_weight(class_weight='balanced',classes=np.array([0,1]), y=y_train)

weights_dict = {0: class_weights[0], 1: class_weights[1]}

#Initialise the model
model = RandomForestClassifier(class_weight=weights_dict, random_state=42)
model.fit(X_train, y_train)

#Predictions

y_pred = model.predict(X_test)

#Save Predictions
pred_df = X_test.copy()
pred_df['Actual'] = y_test.values
pred_df['Predicted'] = y_pred
pred_df.to_csv("data/predictions.csv", index = False)

# EValuation Metrics
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("data/class_report.csv", index=False)
print("Classification Report generated at data/class_report.csv")
#Save the model as pkl file
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_base.pkl")
print("Model Saved to model/rf_base.pkl")