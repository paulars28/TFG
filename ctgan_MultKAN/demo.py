"""Demo module."""

import pandas as pd

DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'


def load_demo():
    #return pd.read_csv(DEMO_URL, compression='gzip')
    #return pd.read_csv("diabetes.csv")
    #return pd.read_csv("EEG_Eye_State_Classification.csv")
    return pd.read_csv("Heart Prediction Quantum Dataset.csv")
    #return pd.read_csv("credit_scoring.csv")


