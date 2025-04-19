# About Project

This project uses payment data to detect fraud. We have fetched data from mySQL, Python and Modeled using sklearn.

# Project Structure

- 'sql/' : SQL queries to fetch data from mySQL
- 'scripts/': Python scripts for data processing and modelling
- 'data/' : contains cleaned data and other outputs
- '.env' : Secure db credentials (not tracked) - included in .gitignore

# Setup
- pip install -r requirements.txt
- python3 scripts/preprocessing.py
- python3 scripts/analysis.py
- python3 scripts/modelling.py