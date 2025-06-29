from dotenv import load_dotenv
load_dotenv()
import os

DB_URL = os.getenv("DB_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")