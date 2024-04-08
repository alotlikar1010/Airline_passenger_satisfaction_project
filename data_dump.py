import pymongo
import pandas as pd
import json

from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://alotlikar98:KqohCK1c3Se5t0yo@cluster0.mqcd2fc.mongodb.net/"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection


DATA_FILE_PATH =(r"C:\Users\91880\Desktop\Airline-satisfaction\Airline_passenger_satisfaction_project\final.csv")

DATABASE_NAME ="airline"

COLLECTION_NAME="collection"


if __name__ =="__main__":
    df =pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    df.reset_index(drop=True ,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())

    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)