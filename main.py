import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import json
import os

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
DB_PATH = "./movie_db"

# --- SETUP ---
# Connect to the same database we created in ingest.py
client = chromadb.PersistentClient(path=DB_PATH)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="movies", embedding_function=sentence_transformer_ef)

# Setup LLM
groq_client = Groq(api_key=GROQ_API_KEY)

def recommend_movie(user_query):
    print(f"\n🔎 Processing: '{user_query}'")
    
    # 1. RETRIEVAL (Vector Search)
    results = collection.query(
        query_texts=[user_query], # Chroma handles embedding automatically here
        n_results=3
    )
    
    # Format candidates for the LLM
    candidates_text = ""
    for i in range(len(results['ids'][0])):
        title = results['metadatas'][0][i]['title']
        overview = results['documents'][0][i]
        candidates_text += f"- Movie: {title}\n  Overview: {overview}\n\n"

    print(f"   (Found {len(results['ids'][0])} candidates via vector search...)")

    # 2. RERANKING (LLM)
    prompt = f"""
    You are a movie recommendation engine.
    User Query: "{user_query}"
    
    Candidates found:
    {candidates_text}
    
    Analyze the candidates. Pick the SINGLE best match for the user's specific "vibe" or request.
    If none are good, say so.
    
    Return ONLY JSON:
    {{
        "selected_movie": "Title of movie",
        "reason": "One short sentence explaining why."
    }}
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-120b",
        response_format={"type": "json_object"}
    )

    return json.loads(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    # Test Loop
    while True:
        user_input = input("\n🎥 Describe the movie vibe you want (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        try:
            recommendation = recommend_movie(user_input)
            print("-" * 40)
            print(f"🎬 BEST PICK: {recommendation['selected_movie']}")
            print(f"💡 REASON:    {recommendation['reason']}")
            print("-" * 40)
        except Exception as e:
            print(f"Error: {e}")