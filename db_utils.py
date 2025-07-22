# db_utils.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, OpenAIError

# Load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PG_CONN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "maitai",
    "user": "postgres",
    "password": os.getenv("PG_PASSWORD")
}

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except (APIConnectionError, APITimeoutError, RateLimitError, OpenAIError) as e:
        return None

def find_similar_inference(embedding, threshold=0.1):
    vector_str = f"[{', '.join(map(str, embedding))}]"
    with psycopg2.connect(**PG_CONN) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT inf.input_text, inf.output_text,
                       emb.embedding <-> %s AS distance
                FROM inference_logs inf
                JOIN embeddings emb ON inf.id = emb.inference_id
                WHERE inf.output_text IS NOT NULL
                ORDER BY emb.embedding <-> %s
                LIMIT 1;
            """, (vector_str, vector_str))
            return cur.fetchone()

def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def save_inference(user_input, model_name, output, embedding, latency_ms=250):
    vector_str = f"[{', '.join(map(str, embedding))}]"
    with psycopg2.connect(**PG_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO inference_logs (user_id, model_name, input_text, output_text, latency_ms, success)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, ('web_user', model_name, user_input, output, latency_ms, True))
            inf_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO embeddings (inference_id, embedding)
                VALUES (%s, %s);
            """, (inf_id, vector_str))
