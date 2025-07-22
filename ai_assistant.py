import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError, APIConnectionError, APITimeoutError

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# PostgreSQL connection
PG_CONN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "maitai",
    "user": "postgres",
    "password": "wtava7as!A"  # Replace if not already in .env
}

def get_embedding(text):
    """Generate 1536-d embedding for the input text."""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except (RateLimitError, APIConnectionError, APITimeoutError, OpenAIError) as e:
        log_error_to_db(None, type(e).__name__, str(e))
        print(f"🔌 Connection or API error: {e}")
        return None

def find_similar_inference(embedding, threshold=0.1):
    """Check for closest match in embeddings table using pgvector."""
    if embedding is None:
        return None

    with psycopg2.connect(**PG_CONN) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
    		SELECT inf.id, inf.input_text, inf.output_text,
           	emb.embedding <-> %s::vector AS distance
    		FROM inference_logs inf
    		JOIN embeddings emb ON inf.id = emb.inference_id
    		ORDER BY emb.embedding <-> %s::vector
    		LIMIT 1;
	"""		
            cur.execute(query, (embedding, embedding))
            result = cur.fetchone()
            if result and result["distance"] < threshold:
                return result
            return None

def generate_response(prompt):
    """Generate response using OpenAI GPT model."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except (RateLimitError, APIConnectionError, APITimeoutError, OpenAIError) as e:
        log_error_to_db(None, type(e).__name__, str(e))
        print(f"💥 Unexpected failure: {e}")
        return None

def save_inference_and_embedding(user_input, model_name, output_text, embedding, latency_ms=250):
    """Insert new record into inference_logs and embeddings tables."""
    if output_text is None or embedding is None:
        # Don't save if we don't have a valid output or embedding
        return

    with psycopg2.connect(**PG_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO inference_logs (user_id, model_name, input_text, output_text, latency_ms, success)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, ('user1', model_name, user_input, output_text, latency_ms, True))
            new_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO embeddings (inference_id, embedding)
                VALUES (%s, %s);
            """, (new_id, embedding))

def log_error_to_db(inference_id, error_type, error_message):
    """Insert error details into error_logs table."""
    with psycopg2.connect(**PG_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO error_logs (inference_id, error_type, error_message)
                VALUES (%s, %s, %s);
            """, (inference_id, error_type, error_message))

def main():
    user_input = input("Ask your question: ").strip()
    print("Generating embedding...")
    embedding = get_embedding(user_input)
    if embedding is None:
        print("❌ Embedding generation failed. Cannot proceed.")
        return

    print("Searching for similar cached responses...")
    match = find_similar_inference(embedding)

    if match:
        print(f"\n✅ Found cached answer:\n{match['output_text']}")
        # Do NOT save new inference log here since it's a cache hit
    else:
        print("\n❌ No cached response. Generating new response...")
        output = generate_response(user_input)
        if output:
            print(f"\n🤖 Generated answer:\n{output}")
            save_inference_and_embedding(user_input, "gpt-3.5-turbo", output, embedding)
        else:
            print("❌ Failed to generate response from the model.")

if __name__ == "__main__":
    main()
