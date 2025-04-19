import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os


#loading the dot env
load_dotenv()

#DB Connection

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

#load fetch_data.sql to read the data from DB
with open("sql/fetch_data.sql","r") as file:
    query = file.read()

#read it as a dataframe
df = pd.read_sql(query, conn)

#Preprocessing
df.dropna(inplace=True)

#Encoding
df_encoded = pd.get_dummies(df,drop_first=True)

df_encoded.to_csv("data/preprocessed_data.csv", index=False)

print("Data Preprocessing complete. Saved to data/preprocessed_data.csv")
