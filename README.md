# PDF Question Answering with Ollama

A local PDF question-answering system powered by Ollama. Upload any PDF and chat with it using AI - completely offline, no API keys required.

> **Note:** This project was built entirely with [Claude Code](https://claude.ai/claude-code) as an exploration of AI-assisted development. I'm just exploring what's possible with AI coding assistants!

## Features

- **100% Local** - All processing happens on your machine
- **No API Keys** - Free to use, no subscriptions
- **Privacy First** - Your documents never leave your computer
- **Modern UI** - Clean Streamlit interface with dark theme
- **Context Viewer** - See which parts of the PDF were used for answers

## Screenshots

The app features a modern dark-themed interface with:
- Sidebar for system status and PDF upload
- Chat interface for asking questions
- Expandable context viewer to see source excerpts

## Requirements

- Windows 10/11
- Python 3.8+
- Docker Desktop
- 10GB+ free disk space (for AI models)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jayaditya-ai/ollama-reader.git
   cd ollama-reader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama via Docker**
   ```bash
   docker-compose up -d
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   Or simply double-click `start.bat` on Windows.

5. **First-time setup**
   - The app will prompt you to download the required AI models
   - Click the download buttons in the sidebar (~2.7GB total)

## Usage

1. Upload a PDF using the sidebar
2. Click "Process PDF" to analyze it
3. Ask questions in the chat input
4. View AI-generated answers with source context

### Example Questions

- "What is this document about?"
- "Summarize the key findings"
- "What does it say about [specific topic]?"
- "List the main conclusions"

## Tech Stack

- **Frontend**: Streamlit
- **LLM Server**: Ollama (Docker)
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDF2
- **Text Splitting**: LangChain
- **Models**:
  - Chat: llama3.2 (~2GB)
  - Embeddings: nomic-embed-text (~700MB)

## How It Works

1. **PDF Upload** - Extract text from uploaded PDF
2. **Chunking** - Split text into manageable pieces
3. **Embedding** - Convert chunks to vectors using Ollama
4. **Storage** - Save vectors in ChromaDB
5. **Query** - Find relevant chunks via similarity search
6. **Answer** - Generate response using local LLM

## Project Structure

```
ollama-reader/
├── app.py              # Main Streamlit application
├── style.css           # Custom dark theme styles
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Ollama Docker configuration
├── start.bat           # Windows launcher script
└── README.md           # This file
```

## Troubleshooting

**Ollama not running?**
```bash
docker-compose up -d
```

**Models missing?**
Click the download buttons in the sidebar when you first run the app.

**Slow responses?**
This is normal for local AI. First questions take longer as models load into memory.

## License

MIT License - feel free to use and modify.

---

*Built with Claude Code - exploring AI-assisted development*
