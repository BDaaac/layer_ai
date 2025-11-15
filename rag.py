"""
RAG (Retrieval-Augmented Generation) pipeline for the AI Lawyer project.
Handles document indexing, embedding generation, and similarity search.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required packages
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available")

from utils import load_law_texts, split_text, clean_text


class RAGPipeline:
    """
    RAG Pipeline for legal document search and retrieval.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    def _load_encoder(self):
        """Load the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available. Please install: pip install sentence-transformers")
            return False
            
        if self.encoder is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                self.encoder = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # Fallback to a smaller model if the main one fails
                logger.info("Trying fallback model...")
                try:
                    self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    return True
                except:
                    logger.error("Failed to load fallback model")
                    return False
        return True
    
    def build_index(self, data_dir: str = "data/laws_raw", 
                   chunk_size: int = 1000, overlap: int = 200,
                   force_rebuild: bool = False) -> bool:
        """
        Build or load the FAISS index from legal documents.
        
        Args:
            data_dir (str): Directory containing legal documents
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            force_rebuild (bool): Force rebuilding even if index exists
            
        Returns:
            bool: Success status
        """
        index_path = "data/faiss.index"
        chunks_path = "data/laws_chunks.pkl"
        
        # Check if required packages are available
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            logger.error("Required packages not available. Please install: pip install sentence-transformers faiss-cpu")
            return False
        
        # Check if index already exists
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
            logger.info("Loading existing index...")
            return self._load_existing_index(index_path, chunks_path)
        
        # Load the encoder
        if not self._load_encoder():
            return False
        
        # Load and process documents
        logger.info("Loading legal documents...")
        law_texts = load_law_texts(data_dir)
        
        if not law_texts:
            logger.warning("No legal documents found. Creating sample document...")
            # Create a sample document for testing
            from utils import get_sample_legal_text
            sample_path = os.path.join(data_dir, "sample_civil_code.txt")
            os.makedirs(data_dir, exist_ok=True)
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(get_sample_legal_text())
            law_texts = load_law_texts(data_dir)
        
        # Process documents into chunks
        self.chunks = []
        self.chunk_metadata = []
        
        for filename, content in law_texts:
            logger.info(f"Processing {filename}...")
            cleaned_content = clean_text(content)
            document_chunks = split_text(cleaned_content, chunk_size, overlap)
            
            for i, chunk in enumerate(document_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'source': filename,
                    'chunk_id': i,
                    'total_chunks': len(document_chunks),
                    'char_count': len(chunk)
                })
        
        logger.info(f"Total chunks created: {len(self.chunks)}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        try:
            embeddings = self.encoder.encode(self.chunks, show_progress_bar=True)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return False
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings.astype(np.float32))
            logger.info(f"Index built with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            return False
        
        # Save index and chunks
        try:
            os.makedirs("data", exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save chunks and metadata
            with open(chunks_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.chunk_metadata
                }, f)
            
            logger.info(f"Index and chunks saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def _load_existing_index(self, index_path: str, chunks_path: str) -> bool:
        """
        Load existing FAISS index and chunks.
        
        Args:
            index_path (str): Path to FAISS index file
            chunks_path (str): Path to chunks pickle file
            
        Returns:
            bool: Success status
        """
        try:
            # Load encoder
            if not self._load_encoder():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_metadata = data['metadata']
            
            logger.info(f"Loaded index with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing index: {str(e)}")
            return False
    
    def search_law(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant legal chunks based on query.
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of relevant chunks with metadata and scores
        """
        if self.index is None or self.encoder is None:
            logger.error("Index or encoder not loaded. Please build index first.")
            return []
        
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Prepare results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = scores[0][i]
                
                if idx < len(self.chunks):
                    result = {
                        'chunk': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': float(score),
                        'rank': i + 1
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dict: Statistics about the RAG pipeline
        """
        if not self.chunks:
            return {"error": "No documents indexed"}
        
        # Calculate statistics
        total_chunks = len(self.chunks)
        total_chars = sum(len(chunk) for chunk in self.chunks)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        # Get unique sources
        sources = list(set(meta['source'] for meta in self.chunk_metadata))
        
        stats = {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': avg_chunk_size,
            'sources': sources,
            'model_name': self.model_name,
            'index_size': self.index.ntotal if self.index else 0
        }
        
        return stats


# Global RAG pipeline instance
rag_pipeline = None

def initialize_rag():
    """Initialize RAG pipeline (call this from Streamlit app)"""
    global rag_pipeline
    if rag_pipeline is None:
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            logger.error("Required packages not available. Please install: pip install sentence-transformers faiss-cpu")
            return False
        
        rag_pipeline = RAGPipeline()
        
        # Try to load existing index
        if os.path.exists("data/faiss.index") and os.path.exists("data/laws_chunks.pkl"):
            logger.info("Loading existing index...")
            success = rag_pipeline._load_existing_index("data/faiss.index", "data/laws_chunks.pkl")
            if success:
                logger.info("Index loaded successfully")
                return True
            else:
                logger.warning("Failed to load existing index")
        
        # If no index exists or loading failed, build new one
        logger.info("Building new index...")
        return build_index(force_rebuild=True)
    return True


def build_index(data_dir: str = "data/laws_raw", force_rebuild: bool = False) -> bool:
    """
    Build the RAG index (convenience function).
    
    Args:
        data_dir (str): Directory containing legal documents
        force_rebuild (bool): Force rebuilding even if index exists
        
    Returns:
        bool: Success status
    """
    global rag_pipeline
    if rag_pipeline is None:
        if not initialize_rag():
            return False
    return rag_pipeline.build_index(data_dir, force_rebuild=force_rebuild)


def search_law(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Search for relevant legal chunks (convenience function).
    
    Args:
        query (str): Search query
        k (int): Number of results to return
        
    Returns:
        List[Dict]: Relevant chunks with metadata
    """
    global rag_pipeline
    if rag_pipeline is None:
        if not initialize_rag():
            return []
    return rag_pipeline.search_law(query, k)


def get_rag_stats() -> Dict[str, Any]:
    """
    Get RAG pipeline statistics (convenience function).
    
    Returns:
        Dict: Pipeline statistics
    """
    global rag_pipeline
    if rag_pipeline is None:
        if not initialize_rag():
            return {}
    return rag_pipeline.get_stats()


if __name__ == "__main__":
    # Test the RAG pipeline
    print("Testing RAG pipeline...")
    
    # Build index
    success = build_index(force_rebuild=True)
    if success:
        print("Index built successfully!")
        
        # Get stats
        stats = get_rag_stats()
        print(f"Pipeline stats: {stats}")
        
        # Test search
        test_queries = [
            "Что такое гражданские права?",
            "Основания возникновения права собственности",
            "Международные договоры и гражданское право"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = search_law(query, k=2)
            
            for i, result in enumerate(results, 1):
                print(f"Result {i} (score: {result['score']:.3f}):")
                print(f"Source: {result['metadata']['source']}")
                print(f"Chunk: {result['chunk'][:200]}...")
                print("-" * 50)
    else:
        print("Failed to build index!")