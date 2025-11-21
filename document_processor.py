"""
Document Processing Module for RAG Application

This module handles PDF parsing, chunking, and multi-modal content processing.
Uses Unstructured library for extraction and Google Gemini for AI summaries.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def parse_pdf(file_path: str) -> List:
    """
    Extract elements from PDF using Unstructured library.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of extracted elements (text, tables, images)
    """
    print(f"📄 Parsing PDF: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",  # High-resolution processing for better accuracy
        infer_table_structure=True,  # Preserve table structure
        extract_image_block_types=["Image"],  # Extract images
        extract_image_block_to_payload=True  # Store images as base64
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements


def create_chunks(elements: List, max_chars: int = 3000) -> List:
    """
    Create intelligent chunks using title-based strategy.
    
    Args:
        elements: List of extracted PDF elements
        max_chars: Maximum characters per chunk
        
    Returns:
        List of chunked elements
    """
    print("🔨 Creating chunks...")
    
    chunks = chunk_by_title(
        elements,
        max_characters=max_chars,
        new_after_n_chars=int(max_chars * 0.8),  # Start new chunk at 80% of max
        combine_text_under_n_chars=500  # Merge small chunks
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk) -> Dict[str, Any]:
    """
    Analyze chunk content and separate by type (text, tables, images).
    
    Args:
        chunk: A single chunk element
        
    Returns:
        Dictionary with separated content types
    """
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_summary(text: str, tables: List[str], images: List[str]) -> str:
    """
    Create AI-enhanced summary for multi-modal content using Gemini.
    
    Args:
        text: Text content
        tables: List of table HTML strings
        images: List of base64 encoded images
        
    Returns:
        AI-generated searchable summary
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Build the prompt
        prompt = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""
        
        # Add tables if present
        if tables:
            prompt += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt += f"Table {i+1}:\n{table}\n\n"
        
        prompt += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed  
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""
        
        # For images, we'll use a simpler approach since Gemini API handles images differently
        # In production, you'd want to use the proper vision API
        if images:
            prompt += f"\n\n[Note: This chunk contains {len(images)} image(s)]"
        
        # Generate summary
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def process_chunks(chunks: List) -> List[Dict[str, Any]]:
    """
    Process all chunks and create AI summaries for multi-modal content.
    
    Args:
        chunks: List of chunked elements
        
    Returns:
        List of processed chunks with metadata
    """
    print("🧠 Processing chunks with AI summaries...")
    
    processed_chunks = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for multi-modal content...")
            enhanced_content = create_ai_summary(
                content_data['text'],
                content_data['tables'], 
                content_data['images']
            )
            print(f"     → AI summary created")
        else:
            enhanced_content = content_data['text']
        
        # Create processed chunk with metadata
        processed_chunk = {
            'content': enhanced_content,
            'metadata': {
                'original_text': content_data['text'],
                'has_tables': len(content_data['tables']) > 0,
                'has_images': len(content_data['images']) > 0,
                'num_tables': len(content_data['tables']),
                'num_images': len(content_data['images']),
                'chunk_index': i
            }
        }
        
        processed_chunks.append(processed_chunk)
    
    print(f"✅ Processed {len(processed_chunks)} chunks")
    return processed_chunks


def process_document(file_path: str, max_chunk_size: int = 3000) -> List[Dict[str, Any]]:
    """
    Main function to process a PDF document end-to-end.
    
    Args:
        file_path: Path to PDF file
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of processed chunks ready for vector storage
    """
    print(f"\n{'='*60}")
    print(f"Processing Document: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")
    
    # Step 1: Parse PDF
    elements = parse_pdf(file_path)
    
    # Step 2: Create chunks
    chunks = create_chunks(elements, max_chars=max_chunk_size)
    
    # Step 3: Process chunks with AI summaries
    processed_chunks = process_chunks(chunks)
    
    print(f"\n{'='*60}")
    print(f"✅ Document processing complete!")
    print(f"Total chunks: {len(processed_chunks)}")
    print(f"{'='*60}\n")
    
    return processed_chunks


if __name__ == "__main__":
    # Test the document processor
    test_file = "AI Enginner Use Case Document.pdf"
    
    if os.path.exists(test_file):
        result = process_document(test_file)
        print(f"\nTest completed: {len(result)} chunks created")
    else:
        print(f"Test file not found: {test_file}")
