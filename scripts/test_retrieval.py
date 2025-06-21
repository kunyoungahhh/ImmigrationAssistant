"""
Utility script to test the retrieval quality of the RAG system.
This helps to tune parameters like chunk size and overlap.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_retrieval(query, index_path="embeddings.index", chunks_path="chunks.txt", metadata_path="metadata.txt", top_k=3):
    """Test retrieval for a given query"""
    try:
        # Load embedding model
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load index
        if not os.path.exists(index_path):
            logger.error(f"Index file not found: {index_path}")
            return False
        
        index = faiss.read_index(index_path)
        
        # Load chunks
        if not os.path.exists(chunks_path):
            logger.error(f"Chunks file not found: {chunks_path}")
            return False
        
        with open(chunks_path, "r") as f:
            chunks = f.read().split("\n===CHUNK_SEPARATOR===\n")
        
        # Load metadata if available
        chunk_metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        page, start = line.strip().split(",")
                        chunk_metadata.append({"page": int(page), "start": int(start)})
        
        # Encode query
        query_vec = embedder.encode([query])
        
        # Search
        k = min(top_k, len(chunks))
        distances, indices = index.search(np.array(query_vec).astype('float32'), k=k)
        
        # Print results
        print(f"\n===== Results for query: '{query}' =====\n")
        
        for i, idx in enumerate(indices[0]):
            print(f"Result {i+1} (distance: {distances[0][i]:.4f}):")
            print("-" * 40)
            print(chunks[idx])
            print("-" * 40)
            
            if chunk_metadata and idx < len(chunk_metadata):
                print(f"Source: Page {chunk_metadata[idx]['page'] + 1}")
            
            print("\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing retrieval: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test retrieval for the RAG system")
    parser.add_argument("query", help="Query to test")
    parser.add_argument("--index", default="embeddings.index", help="Path to the FAISS index file")
    parser.add_argument("--chunks", default="chunks.txt", help="Path to the chunks file")
    parser.add_argument("--metadata", default="metadata.txt", help="Path to the metadata file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    test_retrieval(args.query, args.index, args.chunks, args.metadata, args.top_k)
