"""
Evaluation Module for RAG Application

This module provides evaluation metrics and test question generation.
Implements extra credit features for the interview assessment.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

from rag_pipeline import answer_question, retrieve_context
from vector_store import get_collection_stats

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_test_questions(
    pdf_path: str,
    num_questions: int = 10
) -> List[Dict[str, str]]:
    """
    Generate test questions from a PDF document using Gemini.
    
    Args:
        pdf_path: Path to PDF file
        num_questions: Number of questions to generate
        
    Returns:
        List of question-answer pairs
    """
    print(f"📝 Generating {num_questions} test questions from {pdf_path}...")
    
    # Read a sample of the PDF content (simplified approach)
    # In production, you'd want to process the full document
    from document_processor import parse_pdf
    
    elements = parse_pdf(pdf_path)
    sample_text = " ".join([el.text for el in elements[:50]])[:5000]  # First 5000 chars
    
    prompt = f"""Based on the following document content, generate {num_questions} diverse test questions and their answers.

DOCUMENT CONTENT:
{sample_text}

Generate questions that:
1. Cover different aspects of the document
2. Range from simple factual to complex analytical
3. Can be answered using information in the document

Format your response as a JSON array:
[
    {{"question": "...", "answer": "..."}},
    ...
]

QUESTIONS:"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        questions = json.loads(response_text.strip())
        
        print(f"✅ Generated {len(questions)} test questions")
        return questions
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        return []


def evaluate_retrieval(
    test_questions: List[Dict[str, str]],
    top_k: int = 5,
    collection=None
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance using test questions.
    
    Args:
        test_questions: List of test Q&A pairs
        top_k: Number of chunks to retrieve
        collection: ChromaDB collection
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"🔍 Evaluating retrieval with {len(test_questions)} questions...")
    
    results = []
    
    for qa in test_questions:
        question = qa['question']
        
        # Retrieve context
        retrieved = retrieve_context(question, top_k=top_k, collection=collection)
        
        # Simple relevance check: does retrieved content contain answer keywords?
        answer_words = set(qa['answer'].lower().split())
        retrieved_text = " ".join([chunk['content'].lower() for chunk in retrieved])
        
        # Calculate overlap
        overlap = len([word for word in answer_words if word in retrieved_text])
        relevance_score = overlap / len(answer_words) if answer_words else 0
        
        results.append({
            'question': question,
            'num_retrieved': len(retrieved),
            'relevance_score': relevance_score
        })
    
    # Calculate aggregate metrics
    avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
    
    metrics = {
        'num_questions': len(test_questions),
        'avg_relevance_score': avg_relevance,
        'top_k': top_k,
        'details': results
    }
    
    print(f"✅ Retrieval evaluation complete")
    print(f"   Average relevance: {avg_relevance:.3f}")
    
    return metrics


def evaluate_generation(
    test_questions: List[Dict[str, str]],
    top_k: int = 5,
    collection=None
) -> Dict[str, Any]:
    """
    Evaluate answer generation quality using test questions.
    
    Args:
        test_questions: List of test Q&A pairs
        top_k: Number of chunks to retrieve
        collection: ChromaDB collection
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"🤖 Evaluating answer generation with {len(test_questions)} questions...")
    
    results = []
    
    for qa in test_questions:
        question = qa['question']
        expected_answer = qa['answer']
        
        # Generate answer
        result = answer_question(question, top_k=top_k, collection=collection)
        generated_answer = result['answer']
        
        # Simple quality check: answer length and keyword overlap
        answer_words = set(expected_answer.lower().split())
        generated_words = set(generated_answer.lower().split())
        
        overlap = len(answer_words.intersection(generated_words))
        quality_score = overlap / len(answer_words) if answer_words else 0
        
        results.append({
            'question': question,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'quality_score': quality_score,
            'num_sources': result['num_sources']
        })
    
    # Calculate aggregate metrics
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    
    metrics = {
        'num_questions': len(test_questions),
        'avg_quality_score': avg_quality,
        'top_k': top_k,
        'details': results
    }
    
    print(f"✅ Generation evaluation complete")
    print(f"   Average quality: {avg_quality:.3f}")
    
    return metrics


def compare_search_methods(
    query: str,
    collection=None
) -> Dict[str, Any]:
    """
    Compare similarity search vs semantic search.
    
    Note: This is a simplified comparison. In a full implementation,
    you would implement different search strategies (BM25, hybrid, etc.)
    
    Args:
        query: Test query
        collection: ChromaDB collection
        
    Returns:
        Comparison results
    """
    print(f"⚖️  Comparing search methods for: '{query}'")
    
    # For now, we'll compare different top_k values as a demonstration
    # In production, you'd implement different retrieval strategies
    
    results_k3 = retrieve_context(query, top_k=3, collection=collection)
    results_k5 = retrieve_context(query, top_k=5, collection=collection)
    results_k10 = retrieve_context(query, top_k=10, collection=collection)
    
    comparison = {
        'query': query,
        'methods': {
            'top_k_3': {
                'num_results': len(results_k3),
                'avg_distance': sum(r.get('distance', 0) for r in results_k3) / len(results_k3) if results_k3 else 0
            },
            'top_k_5': {
                'num_results': len(results_k5),
                'avg_distance': sum(r.get('distance', 0) for r in results_k5) / len(results_k5) if results_k5 else 0
            },
            'top_k_10': {
                'num_results': len(results_k10),
                'avg_distance': sum(r.get('distance', 0) for r in results_k10) / len(results_k10) if results_k10 else 0
            }
        }
    }
    
    print(f"✅ Search comparison complete")
    
    return comparison


def run_full_evaluation(
    pdf_path: str,
    collection=None,
    num_test_questions: int = 5
) -> Dict[str, Any]:
    """
    Run complete evaluation suite.
    
    Args:
        pdf_path: Path to test PDF
        collection: ChromaDB collection
        num_test_questions: Number of test questions to generate
        
    Returns:
        Complete evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Running Full Evaluation Suite")
    print(f"{'='*60}\n")
    
    # Generate test questions
    test_questions = generate_test_questions(pdf_path, num_test_questions)
    
    if not test_questions:
        return {'error': 'Failed to generate test questions'}
    
    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(test_questions, collection=collection)
    
    # Evaluate generation
    generation_metrics = evaluate_generation(test_questions, collection=collection)
    
    # Compare search methods
    if test_questions:
        search_comparison = compare_search_methods(
            test_questions[0]['question'],
            collection=collection
        )
    else:
        search_comparison = {}
    
    results = {
        'test_questions': test_questions,
        'retrieval_metrics': retrieval_metrics,
        'generation_metrics': generation_metrics,
        'search_comparison': search_comparison
    }
    
    print(f"\n{'='*60}")
    print(f"✅ Full Evaluation Complete")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Test evaluation module
    from vector_store import initialize_vectorstore
    
    print("Testing Evaluation Module\n")
    
    # Initialize vector store
    collection = initialize_vectorstore()
    
    # Check if we have documents
    stats = get_collection_stats()
    
    if stats['total_chunks'] > 0:
        # Run evaluation
        test_pdf = "AI Enginner Use Case Document.pdf"
        if os.path.exists(test_pdf):
            results = run_full_evaluation(test_pdf, collection, num_test_questions=3)
            
            print("\nEvaluation Summary:")
            print(f"  Retrieval Avg Relevance: {results['retrieval_metrics']['avg_relevance_score']:.3f}")
            print(f"  Generation Avg Quality: {results['generation_metrics']['avg_quality_score']:.3f}")
    else:
        print("No documents in collection. Please process documents first.")
