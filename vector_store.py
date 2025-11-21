"""
Vector Store Module for RAG Application

This module handles vector database operations using ChromaDB and sentence-transformers.
Provides functions for storing and retrieving document chunks.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Global variables for vector store
_client = None
_collection = None
_embedding_model = None


def initialize_vectorstore(
    collection_name: str = "rag_documents",
    persist_directory: str = "./data/chroma_db",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> chromadb.Collection:
    """
    Initialize ChromaDB vector store with sentence-transformers embeddings.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist the database
        embedding_model_name: Name of the sentence-transformers model
        
    Returns:
        ChromaDB collection object
    """
    global _client, _collection, _embedding_model
    
    print(f"🔧 Initializing vector store...")
    print(f"   Collection: {collection_name}")
    print(f"   Embedding model: {embedding_model_name}")
    
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize ChromaDB client
    _client = chromadb.PersistentClient(path=persist_directory)
    
    # Load embedding model
    print(f"   Loading embedding model...")
    _embedding_model = SentenceTransformer(embedding_model_name)
    
    # Get or create collection
    try:
        _collection = _client.get_collection(name=collection_name)
        print(f"   ✅ Loaded existing collection")
    except:
        _collection = _client.create_collection(name=collection_name)
        print(f"   ✅ Created new collection")
    
    return _collection


def add_documents(
    chunks: List[Dict[str, Any]],
    source_file: str,
    collection: Optional[chromadb.Collection] = None
) -> int:
    """
    Add processed document chunks to the vector store.
    
    Args:
        chunks: List of processed chunks with content and metadata
        source_file: Name of the source PDF file
        collection: ChromaDB collection (uses global if not provided)
        
    Returns:
        Number of chunks added
    """
    global _collection, _embedding_model
    
    if collection is None:
        collection = _collection
    
    if collection is None or _embedding_model is None:
        raise ValueError("Vector store not initialized. Call initialize_vectorstore() first.")
    
    print(f"📥 Adding {len(chunks)} chunks to vector store...")
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        # Create unique ID
        chunk_id = f"{source_file}_{i}"
        
        # Prepare metadata
        metadata = {
            'source': source_file,
            'chunk_index': i,
            'has_tables': chunk['metadata'].get('has_tables', False),
            'has_images': chunk['metadata'].get('has_images', False),
            'num_tables': chunk['metadata'].get('num_tables', 0),
            'num_images': chunk['metadata'].get('num_images', 0)
        }
        
        documents.append(chunk['content'])
        metadatas.append(metadata)
        ids.append(chunk_id)
    
    # Generate embeddings
    print(f"   Generating embeddings...")
    embeddings = _embedding_model.encode(documents, show_progress_bar=True)
    
    # Add to collection
    print(f"   Storing in ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search_similar(
    query: str,
    top_k: int = 5,
    collection: Optional[chromadb.Collection] = None,
    filter_metadata: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks using semantic similarity.
    
    Args:
        query: Search query
        top_k: Number of results to return
        collection: ChromaDB collection (uses global if not provided)
        filter_metadata: Optional metadata filters
        
    Returns:
        List of retrieved chunks with content and metadata
    """
    global _collection, _embedding_model
    
    if collection is None:
        collection = _collection
    
    if collection is None or _embedding_model is None:
        raise ValueError("Vector store not initialized. Call initialize_vectorstore() first.")
    
    # Generate query embedding
    query_embedding = _embedding_model.encode([query])[0]
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=filter_metadata
    )
    
    # Format results
    retrieved_chunks = []
    for i in range(len(results['ids'][0])):
        chunk = {
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None
        }
        retrieved_chunks.append(chunk)
    
    return retrieved_chunks


def get_collection_stats(collection: Optional[chromadb.Collection] = None) -> Dict[str, Any]:
    """
    Get statistics about the vector store collection.
    
    Args:
        collection: ChromaDB collection (uses global if not provided)
        
    Returns:
        Dictionary with collection statistics
    """
    global _collection
    
    if collection is None:
        collection = _collection
    
    if collection is None:
        return {"error": "Collection not initialized"}
    
    # Get collection count
    count = collection.count()
    
    # Get unique sources
    all_metadata = collection.get()
    sources = set()
    if all_metadata and 'metadatas' in all_metadata:
        for metadata in all_metadata['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
    
    stats = {
        'total_chunks': count,
        'unique_sources': len(sources),
        'sources': list(sources)
    }
    
    return stats


def delete_document(source_file: str, collection: Optional[chromadb.Collection] = None) -> int:
    """
    Delete all chunks from a specific source document.
    
    Args:
        source_file: Name of the source file to delete
        collection: ChromaDB collection (uses global if not provided)
        
    Returns:
        Number of chunks deleted
    """
    global _collection
    
    if collection is None:
        collection = _collection
    
    if collection is None:
        raise ValueError("Vector store not initialized. Call initialize_vectorstore() first.")
    
    # Get all IDs for this source
    results = collection.get(where={"source": source_file})
    
    if results and 'ids' in results and len(results['ids']) > 0:
        # Delete the chunks
        collection.delete(ids=results['ids'])
        deleted_count = len(results['ids'])
        print(f"🗑️  Deleted {deleted_count} chunks from {source_file}")
        return deleted_count
    
    return 0


def reset_collection(collection_name: str = "rag_documents") -> None:
    """
    Reset (delete and recreate) the collection.
    
    Args:
        collection_name: Name of the collection to reset
    """
    global _client, _collection
    
    if _client is None:
        raise ValueError("Client not initialized. Call initialize_vectorstore() first.")
    
    try:
        _client.delete_collection(name=collection_name)
        print(f"🗑️  Deleted collection: {collection_name}")
    except:
        pass
    
    _collection = _client.create_collection(name=collection_name)
    print(f"✅ Created new collection: {collection_name}")


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store Module\n")
    
    # Initialize
    collection = initialize_vectorstore()
    
    # Get stats
    stats = get_collection_stats()
    print(f"\nCollection Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique sources: {stats['unique_sources']}")
    print(f"  Sources: {stats['sources']}")
