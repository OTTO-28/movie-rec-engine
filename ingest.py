# ingest.py
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup Data (Small sample for demo)
data = [
    {"id": "1", "title": "Lost in Translation", "overview": "Two strangers form an unlikely bond in Tokyo while suffering from insomnia and culture shock."},
    {"id": "2", "title": "The Grand Budapest Hotel", "overview": "A famous concierge at a famous European hotel between the wars and his friendship with a young employee."},
    {"id": "3", "title": "Blade Runner 2049", "overview": "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard."},
    {"id": "4", "title": "Amélie", "overview": "Amélie is an innocent and naive girl in Paris with her own sense of justice. She decides to help those around her."},
    {"id": "5", "title": "Arrival", "overview": "A linguist works with the military to communicate with alien lifeforms after twelve spacecraft appear around the world."},
    {"id": "6", "title": "Whiplash", "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."},
    {"id": "7", "title": "Her", "overview": "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need."}
]

def run_ingestion():
    print("🚀 Starting Data Ingestion...")
    
    # Initialize Persistent Client (Saves data to disk)
    client = chromadb.PersistentClient(path="./movie_db")
    
    # Use a standard open-source embedding model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Get or create collection
    collection = client.get_or_create_collection(name="movies", embedding_function=sentence_transformer_ef)
    
    # Prepare data for Chroma
    ids = [d['id'] for d in data]
    documents = [d['overview'] for d in data]
    metadatas = [{'title': d['title']} for d in data]
    
    # Add to DB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully indexed {len(data)} movies into './movie_db'")

if __name__ == "__main__":
    run_ingestion()