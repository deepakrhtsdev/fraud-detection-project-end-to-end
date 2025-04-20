import numpy as np
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

"""
1. init - get the preprocessed data
2. load_data - read csv, create X and y, train_test_split
3. train - compute_class_weight, weight_dict, model instantiate and fit
4. evaluate and save - prediction, df.copy, combine with prediction (label actual and predicted), save prediction as csv
    classification report, save classification report, save model as pkl under models/....
5. main
"""

class ModelTrainer:

    def __init__(self,data_path="data/preprocessed_data.csv"):
        self.data_path = data_path
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop("label",axis = 1)
        y=df["label"]
        splitted_data = train_test_split(X,y, test_size=0.2, random_state=12)
        return splitted_data

    def train(self,X_train, y_train):
        class_weight = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_train)
        weights_dict = {0: class_weight[0], 1: class_weight[1]}
        print(f"Class Weights: {weights_dict}")
        self.model = RandomForestClassifier(class_weight=weights_dict, random_state=42)
        self.model.fit(X_train, y_train)
        

    def evaluate_and_save(self,X_test, y_test):
        y_pred = self.model.predict(X_test)
        pred_df = X_test.copy()
        pred_df["Actual"] = y_test.values
        pred_df["Prediction"] = y_pred
        pred_df.to_csv("data/predictions.csv", index=False)
        print("predictions generated at data/predictions.csv")

        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        report_df.to_csv("data/classification_report.csv")
        print("Classification Report generated at data/classification_report.csv")

        joblib.dump(self.model, "models/rf_model.pkl")
        print("Model saved successfully at model/rf_model.pkl")
        

if __name__ == "__main__":
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train(X_train,y_train)
    trainer.evaluate_and_save(X_test,y_test)
