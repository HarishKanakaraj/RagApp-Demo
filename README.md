# RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) application for intelligent document question-answering with multi-modal support, built with Streamlit, ChromaDB, and Google Gemini.
 
** Demo Video**
 I have recorded a complete walkthrough demonstrating the application's capabilities:
  Video Link:   https://www.loom.com/share/3bec484fc7584f87b41de7426ca46673
## 🎯 Overview

This application implements a complete RAG pipeline that:
- Processes multi-modal PDF documents (text, tables, images)
- Creates semantic embeddings for intelligent retrieval
- Generates context-aware answers with source citations
- Provides real-time processing statistics
- Includes feedback mechanisms and evaluation metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (recommended) or 3.10
- Windows OS (tested on Windows 11)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Poppler and Tesseract (for PDF processing)

### Installation

1. **Clone or Download the Repository**

2. **Create Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Install External Tools**
   - **Poppler:** Download from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases)
   - **Tesseract:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add both to your System PATH
   - See `EXTERNAL_TOOLS_SETUP.md` for detailed instructions

5. **Configure API Key**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

6. **Run the Application**
   ```powershell
   streamlit run app.py
   ```
   The app will open at http://localhost:8501

## 📋 How to Use

1. **Initialize Vector Store:** Click the button in the sidebar
2. **Upload Documents:** Select PDF files and click "Process Documents"
3. **Ask Questions:** Type your question in the chat interface
4. **View Sources:** Expand the sources section to see citations
5. **Provide Feedback:** Use 👍/👎 buttons to rate answers

## 🏗️ Architecture & Design Decisions

### Embedding Model: Sentence-Transformers (all-MiniLM-L6-v2)

**Why this choice:**
- **Open-source:** No API costs, runs locally
- **Efficient:** 384-dimensional embeddings, fast inference
- **Proven:** Widely used in production RAG systems
- **Semantic understanding:** Captures meaning beyond keywords

**Alternatives considered:**
- OpenAI embeddings (requires API calls, costs)
- Larger models like all-mpnet-base-v2 (slower, marginal gains)

### Vector Database: ChromaDB

**Why this choice:**
- **Simplicity:** Easy to set up and use
- **Persistence:** Built-in disk storage
- **Metadata support:** Rich filtering capabilities
- **Python-native:** Seamless integration

**Alternatives considered:**
- Pinecone (cloud-based, requires account)
- Weaviate (more complex setup)
- FAISS (no metadata support)

### LLM: Google Gemini 2.0 Flash

**Why this choice:**
- **Multi-modal:** Handles text, tables, and images
- **Fast:** Low latency for real-time responses
- **Large context:** 1M token context window
- **Cost-effective:** Generous free tier

**Alternatives considered:**
- GPT-4 (expensive, slower)
- Claude (no free tier)
- Open-source LLMs (require local GPU)

### Chunking Strategy: Title-Based Chunking

**Why this choice:**
- **Semantic coherence:** Keeps related content together
- **Respects structure:** Preserves document hierarchy
- **Optimal size:** 3000 chars max, 2400 chars target
- **Context preservation:** Avoids splitting mid-topic

**Implementation:**
```python
chunk_by_title(
    elements,
    max_characters=3000,
    new_after_n_chars=2400,
    combine_text_under_n_chars=500
)
```

**Why not fixed-size chunks:**
- Fixed chunks can split sentences/paragraphs
- Lose semantic meaning at boundaries
- Title-based respects document structure

### Document Processing: Unstructured Library

**Why this choice:**
- **Multi-modal extraction:** Handles text, tables, images
- **High-resolution strategy:** Better accuracy than basic parsers
- **Table structure preservation:** Maintains HTML table format
- **Image OCR:** Extracts text from images
- **AI-enhanced summaries:** Creates searchable descriptions for complex content

**Processing pipeline:**
1. Parse PDF with `hi_res` strategy
2. Extract tables as structured HTML
3. Extract images as base64
4. Chunk by title for semantic coherence
5. Generate AI summaries for multi-modal chunks
6. Create embeddings for all content

## 🎨 Features

### Core Features
- ✅ Multi-modal document processing (text, tables, images)
- ✅ Semantic search with sentence-transformers
- ✅ Context-aware answer generation with Gemini
- ✅ Source citations with document name and chunk index
- ✅ Persistent vector storage with ChromaDB

### Extra Credit Features
- ✅ **Multi-document retrieval:** Handle multiple PDFs simultaneously
- ✅ **Source citations:** Show exact document and chunk for each answer
- ✅ **Feedback loop:** 👍👎 buttons for answer rating
- ✅ **Search comparison:** Compare different retrieval strategies
- ✅ **Evaluation metrics:** Automated testing and quality measurement

### Advanced Features
- ✅ **Real-time processing statistics:** Shows extraction progress and content breakdown
- ✅ **Content type visualization:** Displays text/table/image distribution
- ✅ **AI-enhanced summaries:** Improves searchability of complex content
- ✅ **Chat history:** Maintains conversation context
- ✅ **Document management:** Upload, view, and delete documents

## 📊 Real-Time Processing Statistics

When uploading documents, the app displays:
- **Progress tracking:** Step-by-step processing status
- **Content metrics:** Total chunks, tables, images extracted
- **Type distribution:** Breakdown of text-only vs multi-modal chunks
- **Visual charts:** Percentage distribution with progress bars

This showcases the power of the Unstructured library and provides transparency into the processing pipeline.

## 🧪 Evaluation

Run automated evaluation:
```powershell
python evaluation.py
```

This generates:
- Test questions from documents
- Retrieval accuracy metrics
- Answer generation quality scores
- Search method comparisons

## 📁 Project Structure

```
Rag App/
├── app.py                      # Main Streamlit application
├── document_processor.py       # PDF parsing and chunking
├── vector_store.py            # ChromaDB operations
├── rag_pipeline.py            # Retrieval and generation
├── evaluation.py              # Metrics and testing
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in repo)
├── README.md                 # This file
└── data/
    ├── uploads/              # Uploaded PDFs
    └── chroma_db/           # Vector database
```

## ⚠️ Limitations

1. **Processing Speed:** High-resolution PDF processing can take 1-2 minutes per document
2. **Memory Usage:** Large documents (>100 pages) may require significant RAM
3. **Image Quality:** OCR accuracy depends on image resolution
4. **Table Complexity:** Very complex tables may not parse perfectly
5. **Context Window:** Very long documents may exceed LLM context limits

## 🔮 Possible Improvements

### Short-term
1. **Hybrid search:** Combine BM25 (keyword) + semantic search
2. **Reranking:** Use cross-encoder for better result ordering
3. **Streaming responses:** Display answers as they generate


### Long-term
1. **Fine-tuning:** Train custom embedding model on domain data
2. **Active learning:** Use feedback to improve retrieval
3. **Graph RAG:** Build knowledge graphs from documents
4. **Agentic workflows:** Multi-step reasoning for complex queries



