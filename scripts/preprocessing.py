import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

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

    def read_query_from_file(self, sql_path = "sql/fetch_data.sql"):
        with open(sql_path, "r") as file:
            query = file.read()

        return query.strip()

    def fetch_data(self,query):
        conn = mysql.connector.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def clean_and_save(self, df, output_path = "data/preprocessed_data.csv"):
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_encoded.to_csv(output_path, index=False)
        print(f"Preprocessing done! Saved to : {output_path}")


if __name__ == "__main__":
    processor = Preprocessor()
    sql_query = processor.read_query_from_file()
    df = processor.fetch_data(sql_query)
    processor.clean_and_save(df)