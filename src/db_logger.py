import psycopg2
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv('./.env')

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "172.16.1.69"),
    "port": os.getenv("PG_PORT", 5432),
    "dbname": os.getenv("PG_DATABASE", "chatbot_db"),
    "user": os.getenv("PG_USER", "chatbot_usr"),
    "password": os.getenv("PG_PASSWORD", "Edutech2025")
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def log_to_db(question: str, answer: str, search_results: list):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        now = datetime.utcnow()

        # Ghi vào bảng ai_response
        cursor.execute("""
            INSERT INTO ai_response (question, metadata, ai_answer, answertime, create_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (question, None, answer, now, now))
        question_id = cursor.fetchone()[0]

        # Ghi các kết quả tìm kiếm vào rag_progress
        for doc in search_results:
            ann_text = doc.page_content
            metadata_json = json.dumps(doc.metadata, ensure_ascii=False)

            cursor.execute("""
                INSERT INTO rag_progress (question_id, ann_return, metadataa)
                VALUES (%s, %s, %s);
            """, (question_id, ann_text, metadata_json))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] {e}")
    finally:
        cursor.close()
        conn.close()

