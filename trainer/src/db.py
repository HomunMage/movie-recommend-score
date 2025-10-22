import os
import psycopg2
from psycopg2.extras import execute_values
import time
import numpy as np

# PostgreSQL Config
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "movielens")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_db_connection(max_retries=5, retry_delay=2):
    """
    Create database connection with retry logic
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
        
    Returns:
        psycopg2 connection object
        
    Raises:
        psycopg2.OperationalError: If connection fails after all retries
    """
    for attempt in range(max_retries):
        try:
            print(f"Connecting to database at {DB_HOST}:{DB_PORT} (attempt {attempt + 1}/{max_retries})")
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=10
            )
            print("✅ Database connection successful")
            return conn
        except psycopg2.OperationalError as e:
            print(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to connect after {max_retries} attempts")
                raise


def wait_for_db(max_attempts=30, delay=2):
    """
    Wait for database to be ready
    
    Args:
        max_attempts: Maximum number of attempts to check database
        delay: Seconds to wait between attempts
        
    Returns:
        bool: True if database is ready, False otherwise
    """
    print("Waiting for database to be ready...")
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=5
            )
            conn.close()
            print("✅ Database is ready!")
            return True
        except psycopg2.OperationalError:
            print(f"Database not ready yet (attempt {attempt + 1}/{max_attempts}), waiting {delay}s...")
            time.sleep(delay)
    
    print("❌ Database failed to become ready in time")
    return False


def save_embeddings_to_db(embeddings, clear_existing=True):
    """
    Save movie embeddings to PostgreSQL with pgvector
    
    Args:
        embeddings: numpy array of shape (num_movies, embedding_dim)
                   Each row is a normalized embedding for movie_id = index + 1
        clear_existing: Whether to clear existing embeddings before inserting
        
    Returns:
        int: Number of embeddings saved
        
    Raises:
        Exception: If database operation fails
    """
    num_movies, embedding_dim = embeddings.shape
    print(f"Saving {num_movies} movie embeddings (dim={embedding_dim}) to database...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear existing embeddings if requested
        if clear_existing:
            cursor.execute("DELETE FROM movie_embeddings")
            print("Cleared existing embeddings from database")
        
        # Prepare data for batch insert
        # movie_id starts from 1 (not 0)
        data = [
            (movie_id + 1, embeddings[movie_id].tolist())
            for movie_id in range(num_movies)
        ]
        
        # Batch insert using execute_values for efficiency
        insert_query = """
            INSERT INTO movie_embeddings (movie_id, embedding)
            VALUES %s
            ON CONFLICT (movie_id) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
        """
        
        execute_values(
            cursor,
            insert_query,
            data,
            template="(%s, %s::vector)"
        )
        
        conn.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM movie_embeddings")
        count = cursor.fetchone()[0]
        print(f"✅ Successfully saved {len(data)} movie embeddings to database")
        print(f"Total embeddings in database: {count}")
        
        cursor.close()
        conn.close()
        
        return count
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        raise


def get_embeddings_from_db(movie_ids=None):
    """
    Retrieve movie embeddings from database
    
    Args:
        movie_ids: List of movie IDs to retrieve. If None, retrieves all.
        
    Returns:
        dict: Dictionary mapping movie_id to embedding array
              e.g., {1: array([...]), 2: array([...]), ...}
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if movie_ids is None:
            query = "SELECT movie_id, embedding FROM movie_embeddings ORDER BY movie_id"
            cursor.execute(query)
        else:
            query = "SELECT movie_id, embedding FROM movie_embeddings WHERE movie_id = ANY(%s) ORDER BY movie_id"
            cursor.execute(query, (movie_ids,))
        
        results = cursor.fetchall()
        
        embeddings_dict = {
            movie_id: np.array(embedding)
            for movie_id, embedding in results
        }
        
        print(f"✅ Retrieved {len(embeddings_dict)} embeddings from database")
        
        cursor.close()
        conn.close()
        
        return embeddings_dict
        
    except Exception as e:
        print(f"❌ Error retrieving embeddings: {e}")
        raise


def get_similar_movies(movie_id, top_k=10):
    """
    Find similar movies using cosine similarity
    
    Args:
        movie_id: The movie ID to find similar movies for
        top_k: Number of similar movies to return
        
    Returns:
        list: List of tuples (movie_id, similarity_score) ordered by similarity
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the target movie's embedding
        cursor.execute(
            "SELECT embedding FROM movie_embeddings WHERE movie_id = %s",
            (movie_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            print(f"❌ Movie {movie_id} not found in database")
            return []
        
        target_embedding = result[0]
        
        # Find similar movies using cosine similarity
        query = """
            SELECT 
                movie_id,
                1 - (embedding <=> %s::vector) as similarity
            FROM movie_embeddings
            WHERE movie_id != %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        cursor.execute(query, (target_embedding, movie_id, target_embedding, top_k))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"✅ Found {len(results)} similar movies for movie {movie_id}")
        return results
        
    except Exception as e:
        print(f"❌ Error finding similar movies: {e}")
        raise


def get_embedding_stats():
    """
    Get statistics about embeddings in the database
    
    Returns:
        dict: Statistics including count, dimension, etc.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM movie_embeddings")
        count = cursor.fetchone()[0]
        
        # Get dimension (from first embedding)
        cursor.execute("SELECT vector_dims(embedding) FROM movie_embeddings LIMIT 1")
        result = cursor.fetchone()
        dimension = result[0] if result else 0
        
        # Get min and max movie_id
        cursor.execute("SELECT MIN(movie_id), MAX(movie_id) FROM movie_embeddings")
        min_id, max_id = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        stats = {
            'count': count,
            'dimension': dimension,
            'min_movie_id': min_id,
            'max_movie_id': max_id
        }
        
        print(f"📊 Database stats: {count} embeddings, dim={dimension}, movie_id range=[{min_id}, {max_id}]")
        return stats
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        raise


def test_connection():
    """
    Test database connection and setup
    
    Returns:
        bool: True if connection and setup are OK
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"PostgreSQL version: {version[:50]}...")
        
        # Check for pgvector extension
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)
        has_vector = cursor.fetchone()[0]
        print(f"pgvector extension: {'✅ Installed' if has_vector else '❌ Not installed'}")
        
        # Check if movie_embeddings table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'movie_embeddings'
            );
        """)
        has_table = cursor.fetchone()[0]
        print(f"movie_embeddings table: {'✅ Exists' if has_table else '❌ Does not exist'}")
        
        cursor.close()
        conn.close()
        
        return has_vector and has_table
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False