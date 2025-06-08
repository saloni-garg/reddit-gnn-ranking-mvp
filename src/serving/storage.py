import numpy as np
import redis
import struct
from typing import Optional
from src.serving.config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL

# Connect to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# --- Compression Utilities ---
def compress_embedding(embedding: np.ndarray) -> bytes:
    """Compress a float32 embedding to float16 and serialize to bytes."""
    arr = embedding.astype(np.float16)
    return arr.tobytes()

def decompress_embedding(data: bytes, dim: int) -> np.ndarray:
    """Decompress bytes to float16 embedding and convert to float32."""
    arr = np.frombuffer(data, dtype=np.float16)
    return arr.astype(np.float32)

# --- Storage Utilities ---
def store_embedding(node_type: str, node_id: int, embedding: np.ndarray, ttl: int = CACHE_TTL):
    """Store compressed embedding in Redis."""
    key = f"embedding:{node_type}:{node_id}"
    data = compress_embedding(embedding)
    redis_client.setex(key, ttl, data)

def load_embedding(node_type: str, node_id: int, dim: int) -> Optional[np.ndarray]:
    """Load and decompress embedding from Redis."""
    key = f"embedding:{node_type}:{node_id}"
    data = redis_client.get(key)
    if data is None:
        return None
    return decompress_embedding(data, dim) 