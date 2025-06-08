import numpy as np
import pytest
from src.serving.storage import store_embedding, load_embedding, compress_embedding, decompress_embedding

def test_compression_decompression():
    """Test that compression and decompression maintain reasonable accuracy."""
    # Create a random embedding
    original = np.random.rand(128).astype(np.float32)
    
    # Compress and decompress
    compressed = compress_embedding(original)
    decompressed = decompress_embedding(compressed, 128)
    
    # Check that values are close (within float16 tolerance)
    assert np.allclose(original, decompressed, atol=1e-3)
    
    # Check that compression actually reduced size
    assert len(compressed) < original.nbytes

def test_store_and_load():
    """Test storing and loading embeddings from Redis."""
    # Create a random embedding
    original = np.random.rand(128).astype(np.float32)
    
    # Store embedding
    store_embedding("test", 42, original)
    
    # Load embedding
    loaded = load_embedding("test", 42, 128)
    
    # Check that values are close
    assert np.allclose(original, loaded, atol=1e-3)

def test_nonexistent_embedding():
    """Test loading a non-existent embedding returns None."""
    loaded = load_embedding("nonexistent", 999, 128)
    assert loaded is None

if __name__ == "__main__":
    pytest.main([__file__]) 