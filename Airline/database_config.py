import pymongo
import pandas as pd
import numpy as np
import json
import os, sys
from dataclasses import dataclass
from Airline.logger import logging


# @dataclass
# class EnvironmentVariable:
#     mongo_db_url:str = os.getenv("MONGO_DB_URL")

mongo_db_url ="mongodb+srv://alotlikar98:KqohCK1c3Se5t0yo@cluster0.mqcd2fc.mongodb.net/"
#env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(mongo_db_url)
logging.info(f"testing url {mongo_db_url} ")
logging.info(f"test {mongo_client}")
print(mongo_db_url)