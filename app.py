"""
RAG Application - Streamlit Interface

A document Q&A system using Retrieval-Augmented Generation.
Upload PDFs, ask questions, and get answers with source citations.
"""

import streamlit as st
import os
from datetime import datetime

# Import our modules
from document_processor import process_document
from vector_store import (
    initialize_vectorstore,
    add_documents,
    get_collection_stats,
    delete_document
)
from rag_pipeline import answer_question


# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()


def init_vector_store():
    """Initialize the vector store if not already done."""
    if st.session_state.collection is None:
        with st.spinner("Initializing vector database..."):
            st.session_state.collection = initialize_vectorstore()
        st.success("✅ Vector database initialized!")


def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file with detailed statistics."""
    # Create uploads directory if it doesn't exist
    os.makedirs("data/uploads", exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join("data/uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create placeholders for real-time updates
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    stats_placeholder = st.empty()
    
    try:
        # Step 1: Parsing
        status_placeholder.info("📄 Step 1/3: Parsing PDF with Unstructured (extracting text, tables, images)...")
        progress_bar.progress(10)
        
        processed_chunks = process_document(file_path)
        progress_bar.progress(60)
        
        # Calculate statistics
        stats = calculate_chunk_statistics(processed_chunks)
        
        # Step 2: Embedding
        status_placeholder.info("🧠 Step 2/3: Generating embeddings with Sentence Transformers...")
        progress_bar.progress(70)
        
        # Add to vector store
        num_added = add_documents(
            processed_chunks,
            uploaded_file.name,
            st.session_state.collection
        )
        progress_bar.progress(90)
        
        # Step 3: Complete
        status_placeholder.success("✅ Step 3/3: Complete!")
        progress_bar.progress(100)
        
        # Track processed file
        st.session_state.processed_files.add(uploaded_file.name)
        
        return True, num_added, stats
        
    except Exception as e:
        status_placeholder.error(f"❌ Error: {str(e)}")
        progress_bar.empty()
        return False, str(e), None


def calculate_chunk_statistics(chunks):
    """Calculate detailed statistics about processed chunks."""
    stats = {
        'total_chunks': len(chunks),
        'text_only': 0,
        'with_tables': 0,
        'with_images': 0,
        'tables_and_images': 0,
        'text_and_tables': 0,
        'text_and_images': 0,
        'total_tables': 0,
        'total_images': 0,
        'avg_chunk_size': 0,
        'chunks_by_type': []
    }
    
    total_chars = 0
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        has_tables = metadata.get('has_tables', False)
        has_images = metadata.get('has_images', False)
        num_tables = metadata.get('num_tables', 0)
        num_images = metadata.get('num_images', 0)
        
        # Count totals
        stats['total_tables'] += num_tables
        stats['total_images'] += num_images
        total_chars += len(chunk.get('content', ''))
        
        # Categorize chunks
        if has_tables and has_images:
            stats['tables_and_images'] += 1
            chunk_type = 'Text + Tables + Images'
        elif has_tables and not has_images:
            stats['text_and_tables'] += 1
            chunk_type = 'Text + Tables'
        elif has_images and not has_tables:
            stats['text_and_images'] += 1
            chunk_type = 'Text + Images'
        else:
            stats['text_only'] += 1
            chunk_type = 'Text Only'
        
        stats['chunks_by_type'].append(chunk_type)
    
    stats['avg_chunk_size'] = total_chars // len(chunks) if chunks else 0
    
    return stats


def display_processing_stats(filename, stats):
    """Display detailed processing statistics in an attractive format."""
    st.subheader(f"📊 Processing Statistics: {filename}")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", stats['total_chunks'])
    with col2:
        st.metric("Tables Extracted", stats['total_tables'])
    with col3:
        st.metric("Images Extracted", stats['total_images'])
    with col4:
        st.metric("Avg Chunk Size", f"{stats['avg_chunk_size']} chars")
    
    st.divider()
    
    # Content type breakdown
    st.markdown("### 🎯 Content Type Distribution")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Detailed breakdown
        st.markdown("**Chunk Breakdown:**")
        st.write(f"📝 Text Only: **{stats['text_only']}** chunks")
        st.write(f"📊 Text + Tables: **{stats['text_and_tables']}** chunks")
        st.write(f"🖼️ Text + Images: **{stats['text_and_images']}** chunks")
        st.write(f"🎨 Text + Tables + Images: **{stats['tables_and_images']}** chunks")
    
    with col2:
        # Visual representation using progress bars
        st.markdown("**Percentage Distribution:**")
        total = stats['total_chunks']
        
        if stats['text_only'] > 0:
            pct = (stats['text_only'] / total) * 100
            st.progress(stats['text_only'] / total, text=f"Text Only: {pct:.1f}%")
        
        if stats['text_and_tables'] > 0:
            pct = (stats['text_and_tables'] / total) * 100
            st.progress(stats['text_and_tables'] / total, text=f"Text + Tables: {pct:.1f}%")
        
        if stats['text_and_images'] > 0:
            pct = (stats['text_and_images'] / total) * 100
            st.progress(stats['text_and_images'] / total, text=f"Text + Images: {pct:.1f}%")
        
        if stats['tables_and_images'] > 0:
            pct = (stats['tables_and_images'] / total) * 100
            st.progress(stats['tables_and_images'] / total, text=f"Multi-modal: {pct:.1f}%")
    
    st.divider()
    
    # Highlight multi-modal capabilities
    if stats['total_tables'] > 0 or stats['total_images'] > 0:
        st.success(f"🎉 **Unstructured Library Power:** Successfully extracted **{stats['total_tables']} tables** and **{stats['total_images']} images** with AI-enhanced summaries for better searchability!")
    else:
        st.info("📄 Document contains primarily text content.")


def display_chat_message(role, content, sources=None, message_idx=None):
    """Display a chat message with optional sources."""
    with st.chat_message(role):
        st.markdown(content)
        
        if sources and role == "assistant":
            with st.expander(f"📚 View Sources ({len(sources)} documents)"):
                for source in sources:
                    st.markdown(f"**Source {source['number']}: {source['source']}** (Chunk {source['chunk_index']})")
                    st.markdown(f"*Relevance Score: {1 - source['distance']:.3f}*" if source['distance'] != 'N/A' else "")
                    
                    # Show content type indicators
                    indicators = []
                    if source['has_tables']:
                        indicators.append("📊 Contains Tables")
                    if source['has_images']:
                        indicators.append("🖼️ Contains Images")
                    if indicators:
                        st.markdown(" | ".join(indicators))
                    
                    # Create unique key using message index and source number
                    unique_key = f"source_{message_idx}_{source['number']}" if message_idx is not None else f"source_{source['number']}"
                    st.text_area(
                        f"Content Preview",
                        source['content'],
                        height=150,
                        key=unique_key,
                        disabled=True
                    )
                    st.divider()


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload documents and ask questions. Get answers with source citations.")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Initialize vector store button
        if st.button("🔧 Initialize Vector Store", use_container_width=True):
            init_vector_store()
        
        st.divider()
        
        # File upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", use_container_width=True):
                # Initialize vector store if needed
                if st.session_state.collection is None:
                    init_vector_store()
                
                # Process each file
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        st.markdown(f"### Processing: {uploaded_file.name}")
                        success, result, stats = process_uploaded_file(uploaded_file)
                        
                        if success:
                            st.success(f"✅ {uploaded_file.name}: {result} chunks added")
                            # Display detailed statistics
                            if stats:
                                display_processing_stats(uploaded_file.name, stats)
                        else:
                            st.error(f"❌ {uploaded_file.name}: {result}")
                    else:
                        st.info(f"ℹ️ {uploaded_file.name} already processed")
        
        st.divider()
        
        # Collection statistics
        st.subheader("📊 Collection Stats")
        if st.session_state.collection is not None:
            stats = get_collection_stats(st.session_state.collection)
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Documents", stats['unique_sources'])
            
            if stats['sources']:
                with st.expander("View Documents"):
                    for source in stats['sources']:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(source)
                        with col2:
                            if st.button("🗑️", key=f"delete_{source}"):
                                delete_document(source, st.session_state.collection)
                                st.rerun()
        else:
            st.info("Initialize vector store to see stats")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many relevant chunks to retrieve for each question"
        )
    
    # Main content area
    if st.session_state.collection is None:
        st.info("👈 Please initialize the vector store from the sidebar to get started.")
        return
    
    stats = get_collection_stats(st.session_state.collection)
    if stats['total_chunks'] == 0:
        st.info("👈 Please upload and process documents from the sidebar to start asking questions.")
        return
    
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        display_chat_message(
            message['role'],
            message['content'],
            message.get('sources'),
            message_idx=i
        )
        
        # Feedback buttons for assistant messages
        if message['role'] == 'assistant':
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"thumbs_up_{i}"):
                    st.session_state.feedback[i] = "positive"
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"thumbs_down_{i}"):
                    st.session_state.feedback[i] = "negative"
                    st.info("Feedback recorded. We'll improve!")
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query
        })
        
        # Display user message
        display_chat_message('user', query, message_idx=len(st.session_state.chat_history)-1)
        
        # Get answer
        with st.spinner("Thinking..."):
            result = answer_question(
                query,
                top_k=top_k,
                collection=st.session_state.collection
            )
        
        # Add assistant message to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result.get('formatted_sources', [])
        })
        
        # Display assistant message
        display_chat_message(
            'assistant',
            result['answer'],
            result.get('formatted_sources', []),
            message_idx=len(st.session_state.chat_history)-1
        )
        
        # Rerun to show feedback buttons
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.feedback = {}
            st.rerun()


if __name__ == "__main__":
    main()
