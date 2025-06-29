import pymysql
import json
from utils.config import DB_CONFIG

def save_analysis(symbol, analysis):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO analysis (symbol, analysis_json, created_at)
                VALUES (%s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                analysis_json=VALUES(analysis_json), created_at=NOW()
            """
            cursor.execute(query, (symbol, json.dumps(analysis)))
            conn.commit()
    finally:
        conn.close()