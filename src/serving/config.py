import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/rgcn_linkpred.pt")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 128))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # Cache time-to-live in seconds

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 