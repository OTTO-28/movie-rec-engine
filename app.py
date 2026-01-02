from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import json
import os
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "./movie_db"
LLM = "openai/gpt-oss-120b"

# --- GLOBAL VARIABLES ---
resources = {}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load heavy ML models and DB connections
    print("⚡ Loading Vector Database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    resources['collection'] = client.get_collection(name="movies", embedding_function=ef)
    
    print("⚡ Connecting to Groq...")
    resources['groq'] = Groq(api_key=GROQ_API_KEY)
    
    yield # The app runs here
    
    # Shutdown: Clean up if needed (not needed for this specific stack)
    print("🛑 Shutting down...")

# Initialize App with Lifespan
app = FastAPI(title="VibeCheck RecSys API", lifespan=lifespan)

# --- PYDANTIC MODELS (Data Validation) ---
class UserQuery(BaseModel):
    query: str

class MovieRecommendation(BaseModel):
    selected_movie: str
    reason: str

class APIResponse(BaseModel):
    user_query: str
    recommendation: MovieRecommendation
    debug_candidates: list[str] # Optional: return list of candidate titles

# --- ROUTES ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "FastAPI + LLM"
    }

@app.post("/recommend", response_model=APIResponse)
def recommend(request: UserQuery):
    """
    Receives a user query (e.g., "Sad movies in space"), 
    retrieves candidates via Vector Search, 
    and reranks them using LLM.
    """
    try:
        user_query = request.query
        collection = resources['collection']
        groq_client = resources['groq']

        # 1. RETRIEVAL (Vector Search)
        results = collection.query(
            query_texts=[user_query],
            n_results=5
        )
        
        # Check if we found anything
        if not results['ids'][0]:
            raise HTTPException(status_code=404, detail="No relevant movies found.")

        # Prepare context for LLM
        candidates_text = ""
        candidates_list = []
        
        for i in range(len(results['ids'][0])):
            title = results['metadatas'][0][i]['title']
            overview = results['documents'][0][i]
            candidates_text += f"- Title: {title}\n  Overview: {overview}\n\n"
            candidates_list.append(title)

        # 2. RERANKING (LLM)
        prompt = f"""
        You are a movie API. User Query: "{user_query}"
        
        Candidates from Vector DB:
        {candidates_text}
        
        Task: Pick the best match for the user's request.
        Return JSON: {{ "selected_movie": "Title", "reason": "Short explanation" }}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM,
            response_format={"type": "json_object"}
        )

        llm_response = json.loads(chat_completion.choices[0].message.content)

        return APIResponse(
            user_query=user_query,
            recommendation=llm_response,
            debug_candidates=candidates_list
        )

    except Exception as e:
        # In production, log the error here
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)