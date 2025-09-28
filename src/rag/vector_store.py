"""
Vector Database Integration

This module provides functionality to store and retrieve embeddings using
both FAISS and ChromaDB for the RAG system.
"""

import os
import json
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
from dataclasses import dataclass
import pickle

from ..database.connection import get_db_context
from ..database.models import VectorMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document class for storing content and metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class EmbeddingModel:
    """Wrapper for sentence transformer embedding models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dimension: {self.dimension})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        embeddings = self.model.encode(texts)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        return self.model.encode([text])[0]


class FAISSVectorStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or os.getenv('FAISS_INDEX_PATH', './data/faiss_index')
        self.index = None
        self.documents = {}  # document_id -> Document
        self.id_to_index = {}  # document_id -> faiss_index
        self.index_to_id = {}  # faiss_index -> document_id
        self.next_index = 0
        
        self._initialize_index()
        self._load_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Use IndexFlatIP for cosine similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the FAISS index"""
        if not documents:
            return
        
        # Extract embeddings and normalize them for cosine similarity
        embeddings = np.array([doc.embedding for doc in documents]).astype('float32')
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Update mappings
        for i, doc in enumerate(documents):
            faiss_idx = self.next_index + i
            self.documents[doc.id] = doc
            self.id_to_index[doc.id] = faiss_idx
            self.index_to_id[faiss_idx] = doc.id
        
        self.next_index += len(documents)
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                doc_id = self.index_to_id.get(idx)
                if doc_id and doc_id in self.documents:
                    results.append((self.documents[doc_id], float(score)))
        
        return results
    
    def save_index(self):
        """Save FAISS index to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save metadata
        metadata = {
            'documents': {doc_id: {
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata
            } for doc_id, doc in self.documents.items()},
            'id_to_index': self.id_to_index,
            'index_to_id': {str(k): v for k, v in self.index_to_id.items()},
            'next_index': self.next_index,
            'dimension': self.dimension
        }
        
        with open(f"{self.index_path}.metadata", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def _load_index(self):
        """Load FAISS index from disk"""
        faiss_file = f"{self.index_path}.faiss"
        metadata_file = f"{self.index_path}.metadata"
        
        if os.path.exists(faiss_file) and os.path.exists(metadata_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(faiss_file)
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Restore documents
                for doc_id, doc_data in metadata['documents'].items():
                    self.documents[doc_id] = Document(
                        id=doc_data['id'],
                        content=doc_data['content'],
                        metadata=doc_data['metadata']
                    )
                
                self.id_to_index = metadata['id_to_index']
                self.index_to_id = {int(k): v for k, v in metadata['index_to_id'].items()}
                self.next_index = metadata['next_index']
                
                logger.info(f"Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
                
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self._initialize_index()


class ChromaVectorStore:
    """ChromaDB-based vector store for metadata and document storage"""
    
    def __init__(self, collection_name: str = "argo_profiles", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv('CHROMA_PERSIST_DIRECTORY', './data/chroma_db')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to ChromaDB"""
        if not documents:
            return
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        embeddings = [doc.embedding.tolist() for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents with optional filtering"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_dict
        )
        
        documents = []
        if results['ids'][0]:  # Check if we got results
            for i, doc_id in enumerate(results['ids'][0]):
                doc = Document(
                    id=doc_id,
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                distance = results['distances'][0][i]
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                similarity = 1 / (1 + distance)
                documents.append((doc, similarity))
        
        return documents
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        self.collection.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
    
    def count(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()


class VectorStore:
    """Combined vector store using both FAISS and ChromaDB"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 use_faiss: bool = True,
                 use_chroma: bool = True):
        
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.use_faiss = use_faiss
        self.use_chroma = use_chroma
        
        # Initialize vector stores
        if use_faiss:
            self.faiss_store = FAISSVectorStore(self.embedding_model.dimension)
        else:
            self.faiss_store = None
            
        if use_chroma:
            self.chroma_store = ChromaVectorStore()
        else:
            self.chroma_store = None
    
    def add_documents(self, texts: List[str], metadatas: List[Dict], 
                     document_ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to vector stores"""
        if not document_ids:
            document_ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Create document objects
        documents = []
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, document_ids)):
            doc = Document(
                id=doc_id,
                content=text,
                metadata=metadata,
                embedding=embeddings[i]
            )
            documents.append(doc)
        
        # Add to vector stores
        if self.faiss_store:
            self.faiss_store.add_documents(documents)
        
        if self.chroma_store:
            self.chroma_store.add_documents(documents)
        
        # Store metadata in PostgreSQL
        self._store_vector_metadata(documents)
        
        return document_ids
    
    def search(self, query: str, k: int = 10, 
               use_faiss: bool = None, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode_single(query)
        
        # Use FAISS by default if available
        if use_faiss is None:
            use_faiss = self.faiss_store is not None
        
        if use_faiss and self.faiss_store:
            return self.faiss_store.search(query_embedding, k)
        elif self.chroma_store:
            return self.chroma_store.search(query_embedding, k, filter_dict)
        else:
            return []
    
    def _store_vector_metadata(self, documents: List[Document]):
        """Store vector metadata in PostgreSQL"""
        try:
            with get_db_context() as session:
                for doc in documents:
                    # Check if already exists
                    existing = session.query(VectorMetadata).filter_by(vector_id=doc.id).first()
                    
                    if not existing:
                        vector_meta = VectorMetadata(
                            vector_id=doc.id,
                            document_type=doc.metadata.get('document_type', 'profile_summary'),
                            profile_id=doc.metadata.get('profile_id'),
                            platform_number=doc.metadata.get('platform_number'),
                            embedding_model=self.embedding_model.model_name,
                            vector_dimension=self.embedding_model.dimension,
                            content_summary=doc.content[:500],  # First 500 chars
                            content_metadata=doc.metadata
                        )
                        session.add(vector_meta)
                
                session.commit()
                logger.info(f"Stored metadata for {len(documents)} vectors in PostgreSQL")
                
        except Exception as e:
            logger.error(f"Failed to store vector metadata: {e}")
    
    def save(self):
        """Save vector stores to disk"""
        if self.faiss_store:
            self.faiss_store.save_index()
        
        # ChromaDB automatically persists
        logger.info("Vector stores saved successfully")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector stores"""
        stats = {
            'embedding_model': self.embedding_model.model_name,
            'embedding_dimension': self.embedding_model.dimension,
            'faiss_enabled': self.faiss_store is not None,
            'chroma_enabled': self.chroma_store is not None
        }
        
        if self.faiss_store:
            stats['faiss_count'] = self.faiss_store.index.ntotal
        
        if self.chroma_store:
            stats['chroma_count'] = self.chroma_store.count()
        
        return stats


# Global vector store instance
vector_store = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store


def initialize_vector_store(embedding_model: str = "all-MiniLM-L6-v2"):
    """Initialize the global vector store"""
    global vector_store
    vector_store = VectorStore(embedding_model_name=embedding_model)
    return vector_store


if __name__ == "__main__":
    # Test vector store functionality
    vs = VectorStore()
    
    # Test documents
    texts = [
        "Temperature profile from ARGO float 123456 in the North Atlantic",
        "Salinity measurements from the Arabian Sea collected in March 2023",
        "BGC parameters including oxygen and chlorophyll from equatorial Pacific"
    ]
    
    metadatas = [
        {'profile_id': '123456_001', 'platform_number': '123456', 'document_type': 'profile_summary'},
        {'profile_id': '789012_002', 'platform_number': '789012', 'document_type': 'profile_summary'},
        {'profile_id': '345678_003', 'platform_number': '345678', 'document_type': 'profile_summary'}
    ]
    
    # Add documents
    doc_ids = vs.add_documents(texts, metadatas)
    print(f"Added documents with IDs: {doc_ids}")
    
    # Search
    results = vs.search("temperature measurements from North Atlantic", k=2)
    print(f"Search results: {len(results)}")
    for doc, score in results:
        print(f"  Score: {score:.3f}, Content: {doc.content[:50]}...")
    
    # Save
    vs.save()
    print("Vector store saved successfully")
