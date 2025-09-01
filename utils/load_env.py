# Create config/env_loader.py
import os
from dotenv import load_dotenv

load_dotenv()

ASTRA_CONFIG = {
    'endpoint': os.getenv('ASTRA_DB_ENDPOINT'),
    'token': os.getenv('ASTRA_DB_TOKEN'),
    'keyspace': os.getenv('ASTRA_KEYSPACE', 'memory_db')
}