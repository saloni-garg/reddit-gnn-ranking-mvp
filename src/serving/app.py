from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict, Optional
import redis
import json
import logging

from src.models.rgcn import HeteroRGCN
from src.data.pyg_graph_conversion import load_hetero_data
from src.serving.config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    MODEL_PATH, EMBEDDING_DIM,
    API_HOST, API_PORT,
    CACHE_TTL, LOG_LEVEL
)
from src.serving.storage import store_embedding, load_embedding

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="Reddit Graph Embedding API")

# Load model and data
logger.info("Loading model and data...")
data = load_hetero_data()
model = HeteroRGCN(
    metadata=data.metadata(),
    hidden_channels=EMBEDDING_DIM
)

# Load state dict and handle the 'model.' prefix
state_dict = torch.load(MODEL_PATH)
if all(k.startswith('model.') for k in state_dict.keys()):
    # Remove 'model.' prefix from all keys
    state_dict = {k[6:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
logger.info("Model and data loaded successfully")

# Initialize Redis for caching
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

class EmbeddingRequest(BaseModel):
    node_type: str
    node_ids: List[int]

class EmbeddingStorageRequest(BaseModel):
    node_type: str
    node_id: int
    embedding: List[float]

class RankingRequest(BaseModel):
    user_id: int
    candidate_post_ids: List[int]

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    content_type: str = "post"  # or "subreddit"

def rank_items_for_user(user_id: int, candidate_post_ids: List[int]) -> List[int]:
    """Rank candidate posts for a user based on embedding similarity."""
    try:
        # Get user embedding
        user_emb = load_embedding("user", user_id, EMBEDDING_DIM)
        if user_emb is None:
            raise HTTPException(
                status_code=404,
                detail=f"User embedding not found for user_id: {user_id}"
            )
        
        # Get post embeddings
        post_embs = [load_embedding("post", pid, EMBEDDING_DIM) for pid in candidate_post_ids]
        
        # Skip missing embeddings
        valid = [(pid, emb) for pid, emb in zip(candidate_post_ids, post_embs) if emb is not None]
        if not valid:
            raise HTTPException(
                status_code=404,
                detail="No valid post embeddings found"
            )
        
        # Compute similarity scores
        scores = [(pid, np.dot(user_emb, emb)) for pid, emb in valid]
        
        # Sort by score in descending order
        ranked = [pid for pid, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
        return ranked
    except Exception as e:
        logger.error(f"Error ranking items: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Reddit Graph Embedding API"}

@app.post("/store_embedding")
async def store_embedding_endpoint(request: EmbeddingStorageRequest):
    """Store a single embedding in Redis."""
    try:
        # Convert list to numpy array
        embedding = np.array(request.embedding, dtype=np.float32)
        
        # Validate embedding dimension
        if embedding.shape[0] != EMBEDDING_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension must be {EMBEDDING_DIM}, got {embedding.shape[0]}"
            )
        
        # Store embedding
        store_embedding(request.node_type, request.node_id, embedding)
        logger.info(f"Stored embedding for {request.node_type}:{request.node_id}")
        
        return {"status": "success", "message": "Embedding stored successfully"}
    except Exception as e:
        logger.error(f"Error storing embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_embedding/{node_type}/{node_id}")
async def get_embedding_endpoint(node_type: str, node_id: int):
    """Retrieve a single embedding from Redis."""
    try:
        embedding = load_embedding(node_type, node_id, EMBEDDING_DIM)
        if embedding is None:
            raise HTTPException(
                status_code=404,
                detail=f"Embedding not found for {node_type}:{node_id}"
            )
        
        return {"embedding": embedding.tolist()}
    except Exception as e:
        logger.error(f"Error retrieving embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank_items")
async def rank_items_endpoint(request: RankingRequest):
    """Rank candidate posts for a user based on embedding similarity."""
    try:
        ranked = rank_items_for_user(request.user_id, request.candidate_post_ids)
        return {
            "user_id": request.user_id,
            "ranked_post_ids": ranked
        }
    except Exception as e:
        logger.error(f"Error in rank_items endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """Get embeddings for a list of nodes of a specific type."""
    try:
        # Check if embeddings are in cache
        cache_key = f"embeddings:{request.node_type}:{','.join(map(str, request.node_ids))}"
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached)

        logger.info(f"Computing embeddings for {len(request.node_ids)} {request.node_type} nodes")
        # Get embeddings from model
        with torch.no_grad():
            x_dict = {ntype: data[ntype].x for ntype in data.node_types}
            edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
            edge_type_dict = {etype: data[etype].edge_type for etype in data.edge_types if hasattr(data[etype], 'edge_type')}
            
            embeddings = model(x_dict, edge_index_dict, edge_type_dict)
            node_embeddings = embeddings[request.node_type][request.node_ids].numpy().tolist()

        # Cache the results
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(node_embeddings))
        logger.info(f"Cached embeddings for {cache_key}")
        
        return {"embeddings": node_embeddings}
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get content recommendations for a user."""
    try:
        logger.info(f"Getting recommendations for user {request.user_id}")
        # Get user embedding
        user_embedding = await get_embeddings(EmbeddingRequest(node_type="user", node_ids=[request.user_id]))
        user_embedding = torch.tensor(user_embedding["embeddings"][0])

        # Get content embeddings
        content_type = request.content_type
        content_ids = list(range(data[content_type].x.size(0)))
        content_embeddings = await get_embeddings(EmbeddingRequest(node_type=content_type, node_ids=content_ids))
        content_embeddings = torch.tensor(content_embeddings["embeddings"])

        # Compute similarity scores
        scores = torch.matmul(content_embeddings, user_embedding)
        
        # Get top-k recommendations
        top_k_scores, top_k_indices = torch.topk(scores, min(request.num_recommendations, len(scores)))
        
        recommendations = [
            {"id": int(idx), "score": float(score)}
            for idx, score in zip(top_k_indices, top_k_scores)
        ]
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {request.user_id}")
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT) 