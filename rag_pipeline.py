"""
RAG Pipeline Module

This module implements the Retrieval-Augmented Generation pipeline.
Handles context retrieval and answer generation using Google Gemini.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

from vector_store import search_similar

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def retrieve_context(
    query: str,
    top_k: int = 5,
    collection=None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context chunks for a query.
    
    Args:
        query: User question
        top_k: Number of chunks to retrieve
        collection: ChromaDB collection (optional)
        
    Returns:
        List of retrieved chunks with metadata
    """
    print(f"🔍 Retrieving context for: '{query}'")
    print(f"   Top K: {top_k}")
    
    # Search for similar chunks
    results = search_similar(query, top_k=top_k, collection=collection)
    
    print(f"✅ Retrieved {len(results)} chunks")
    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into context string for LLM.
    
    Args:
        chunks: List of retrieved chunks
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        source = chunk['metadata'].get('source', 'Unknown')
        chunk_idx = chunk['metadata'].get('chunk_index', i)
        content = chunk['content']
        
        context_part = f"""
[Source {i+1}: {source}, Chunk {chunk_idx}]
{content}
"""
        context_parts.append(context_part)
    
    return "\n".join(context_parts)


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    model_name: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Generate answer using retrieved context and Gemini.
    
    Args:
        query: User question
        context_chunks: Retrieved context chunks
        model_name: Gemini model to use
        
    Returns:
        Dictionary with answer and metadata
    """
    print(f"🤖 Generating answer with {model_name}...")
    
    # Format context
    context = format_context(context_chunks)
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant answering questions based on provided document context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sources when making claims (e.g., "According to Source 1...")
4. Be concise but comprehensive
5. If you mention data or statistics, include the source

ANSWER:"""
    
    # Generate answer
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        answer_text = response.text
        
        print(f"✅ Answer generated")
        
        return {
            'answer': answer_text,
            'sources': context_chunks,
            'num_sources': len(context_chunks),
            'query': query
        }
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': context_chunks,
            'num_sources': len(context_chunks),
            'query': query,
            'error': str(e)
        }


def format_sources_for_display(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format source chunks for display in UI.
    
    Args:
        chunks: List of retrieved chunks
        
    Returns:
        List of formatted source dictionaries
    """
    formatted_sources = []
    
    for i, chunk in enumerate(chunks):
        source_info = {
            'number': i + 1,
            'source': chunk['metadata'].get('source', 'Unknown'),
            'chunk_index': chunk['metadata'].get('chunk_index', 'N/A'),
            'content': chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'],
            'full_content': chunk['content'],
            'has_tables': chunk['metadata'].get('has_tables', False),
            'has_images': chunk['metadata'].get('has_images', False),
            'distance': chunk.get('distance', 'N/A')
        }
        formatted_sources.append(source_info)
    
    return formatted_sources


def answer_question(
    query: str,
    top_k: int = 5,
    collection=None
) -> Dict[str, Any]:
    """
    Main RAG pipeline function: retrieve context and generate answer.
    
    Args:
        query: User question
        top_k: Number of context chunks to retrieve
        collection: ChromaDB collection (optional)
        
    Returns:
        Complete response with answer and sources
    """
    print(f"\n{'='*60}")
    print(f"RAG Pipeline: Answering Question")
    print(f"{'='*60}\n")
    
    # Step 1: Retrieve context
    context_chunks = retrieve_context(query, top_k=top_k, collection=collection)
    
    if not context_chunks:
        return {
            'answer': "I couldn't find any relevant information in the documents to answer this question.",
            'sources': [],
            'num_sources': 0,
            'query': query
        }
    
    # Step 2: Generate answer
    result = generate_answer(query, context_chunks)
    
    # Step 3: Format sources for display
    result['formatted_sources'] = format_sources_for_display(context_chunks)
    
    print(f"\n{'='*60}")
    print(f"✅ RAG Pipeline Complete")
    print(f"{'='*60}\n")
    
    return result


if __name__ == "__main__":
    # Test the RAG pipeline
    from vector_store import initialize_vectorstore, get_collection_stats
    
    print("Testing RAG Pipeline Module\n")
    
    # Initialize vector store
    collection = initialize_vectorstore()
    
    # Check if we have documents
    stats = get_collection_stats()
    print(f"Collection has {stats['total_chunks']} chunks")
    
    if stats['total_chunks'] > 0:
        # Test query
        test_query = "What is this document about?"
        result = answer_question(test_query, top_k=3)
        
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources used: {result['num_sources']}")
    else:
        print("\nNo documents in collection. Please add documents first.")
