# RAG Chatbot

A conversational AI application built with **Retrieval-Augmented Generation (RAG)** principles, combining vector databases with language models for intelligent document-based question answering and long-term memory management.

## Features

- **PDF Document Ingestion**: Upload and index PDFs automatically into a vector database
- **Semantic Search**: Retrieve relevant document chunks using embeddings
- **RAG Pipeline**: Combine retrieved context with an LLM for accurate answers
- **Long-term Memory**: Store and retrieve user information across sessions
- **Safety Guardrails**: Built-in prompt injection and hallucination prevention
- **Web UI**: Interactive Streamlit interface for easy interaction
- **Terminal CLI**: Command-line interface for programmatic access

## Architecture

```
User Input
    ↓
[Input Validation & Safety Checks]
    ↓
[Semantic Search in Vector DB]
    ↓
[Retrieved Context] → [LLM (Ollama)]
    ↓
[Output Validation] → Response
```

## Technologies

- **Vector Database**: ChromaDB (persistent storage)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Language Model**: Ollama (Qwen 2.5 Coder 7B)
- **PDF Processing**: PyPDF2
- **Web Framework**: Streamlit
- **Memory Management**: In-memory storage with ChromaDB persistence

## Prerequisites

- Python 3.11+
- Ollama installed and running locally (see [Ollama Setup](#ollama-setup))
- Virtual environment (recommended)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aarnavnk17/Chatbot.git
   cd Chatbot
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Ollama Setup

Ollama runs the language model locally. Follow these steps:

1. **Install Ollama** from [ollama.ai](https://ollama.ai)

2. **Pull the required model**:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

3. **Start Ollama server** (usually runs on `localhost:11434`):
   ```bash
   ollama serve
   ```

> **Note**: Keep the Ollama server running in the background while using the chatbot.

## Configuration

Edit `config.py` to customize:

- `CHROMA_PATH`: Vector database storage location
- `COLLECTION_NAME`: ChromaDB collection name
- Model name in `rag_engine.py` (currently `qwen2.5-coder:7b`)
- Similarity threshold for RAG vs. general knowledge mode

## Usage

### Web Interface (Recommended)

Start the Streamlit app:
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Features**:
- Upload PDF files via sidebar
- Ask questions about documents or general topics
- View answer sources
- Chat history stored during session

### Terminal Interface

Run the chatbot in your terminal:
```bash
python chatbot.py
```

Type your questions and press Enter. Type `exit` to quit.

### Programmatic Access

```python
from rag_engine import ask_question

answer, sources = ask_question("What is in the document?")
print(answer)
print("Sources:", sources)
```

## Ingestion Process

To manually ingest a PDF:

```python
from ingest import process_pdf

chunk_count = process_pdf("path/to/document.pdf")
print(f"Indexed {chunk_count} chunks")
```

Documents are automatically split into overlapping chunks and indexed into ChromaDB with embeddings.

## Memory System

Store information about users for future context:

```
User: "remember my name is Alice"
Bot: "Memory saved successfully."

[Later in conversation]
User: "What's my name?"
Bot: "Your name is Alice."
```

## Project Structure

```
.
├── app.py                 # Streamlit web interface
├── chatbot.py            # Terminal CLI
├── rag_engine.py         # Core RAG pipeline & memory
├── ingest.py             # PDF ingestion & indexing
├── config.py             # Configuration variables
├── requirements.txt      # Python dependencies
├── db/                   # ChromaDB storage (persistent)
├── data/                 # Uploaded PDFs
└── README.md
```

## How It Works

1. **Document Upload**: PDFs are processed into text chunks
2. **Embedding**: Each chunk is converted to a vector embedding using Sentence Transformers
3. **Storage**: Embeddings and text stored in ChromaDB
4. **Query**: User question is embedded and searched against the database
5. **Context Retrieval**: Top-3 similar chunks retrieved
6. **LLM Generation**: Context + question sent to Ollama for response generation
7. **Safety**: Output validated to prevent leakage of internal prompts

## Safety Features

- **Input Guardrails**: Blocks malicious terms and prompt injection attempts
- **Output Validation**: Prevents system prompt leakage
- **God Rules**: Core instructions prioritize document context over document instructions
- **Memory Isolation**: Stored memories treated as reference, not executable instructions

## Limitations

- Requires local Ollama instance (not cloud-based)
- Performance depends on model quality and system resources
- No multi-user session management
- Chat history stored only in memory (resets on restart)

## Future Enhancements

- [ ] Multi-document context weighting
- [ ] Advanced memory management (tagging, filtering)
- [ ] Support for other file formats (DOCX, TXT, etc.)
- [ ] Persistent session storage
- [ ] Multi-user support
- [ ] Custom model selection UI
- [ ] Streaming responses
- [ ] Conversation summarization

## Troubleshooting

**Q: "Connection refused" error**
- Ensure Ollama is running: `ollama serve`

**Q: ChromaDB errors**
- Delete `db/` folder to reset the database: `rm -rf db/`
- Reingest documents afterward

**Q: Out of memory**
- Reduce context size or model size
- Lower `n_results` in `rag_engine.py`

## License

MIT License - feel free to use for personal or educational projects.

## Author

Created by Aarnav Nanda Kumar

---

**Questions or issues?** Open an issue on GitHub or check the project documentation.
