import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
# Aim is to do the preprocessing
"""
1. getting db config using dotenv > getenv
2. read query from sql/fetch_data.sql
3. fetch data from db
4. cleaning and saving as csv to data/preprocessed_data.csv

TO DO: create docstring for class and methods. ALso, try to include more preprocessing steps.
"""

class Preprocessor:

    def __init__(self):
        load_dotenv()
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME")
        }
        self.encoder = OneHotEncoder(drop="first", handle_unknown="ignore")



    def read_query_from_file(self, sql_path = "sql/fetch_data.sql"):
        with open(sql_path, "r") as file:
            query = file.read()

        return query.strip()

    def fetch_data(self,query):
        conn = mysql.connector.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def clean_and_save(self, df, output_path = "data/preprocessed_data.csv",encoder_path = "models/encoder.pkl"):
        categorical_cols = ["paymentMethod"]

        df_encoded = df.copy()


        encoded_arr = self.encoder.fit_transform(df_encoded[categorical_cols]).toarray()
        encoded_df = pd.DataFrame(encoded_arr, columns = self.encoder.get_feature_names_out(categorical_cols), index = df_encoded.index)
        df_encoded.drop(columns=categorical_cols, inplace=True)
        df_encoded =pd.concat([df_encoded.reset_index(drop=True),encoded_df.reset_index(drop=True)], axis=1)
        
        df_encoded.to_csv(output_path, index=False)
        print(f"Preprocessing done! Saved to : {output_path}")

        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoder,f)

        print(f"Encoder saved at {encoder_path}")


if __name__ == "__main__":
    processor = Preprocessor()
    sql_query = processor.read_query_from_file()
    df = processor.fetch_data(sql_query)
    processor.clean_and_save(df)