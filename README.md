# FUTA Knowledge Assistant

A conversational AI chatbot for the Federal University of Technology Akure (FUTA) using Gemini embeddings and semantic search. Available as both a CLI and web application.

## Features

- 🤖 **Conversational AI**: Natural language interaction with memory
- 🔍 **Semantic Search**: Find relevant information using embeddings
- 📚 **Knowledge Base**: Comprehensive FUTA information
- 💬 **Memory**: Remembers conversation context
- 🚀 **Simple Setup**: No complex database requirements
- 🌐 **Web Interface**: Beautiful, responsive web UI
- 📱 **Mobile Friendly**: Works on all devices

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key
Create a `.env` file in the project directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Create Embeddings
```bash
python futa_embedding_creator_simple.py
```

### 4. Choose Your Interface

#### Option A: Web Interface (Recommended)
```bash
python app.py
```
Then open http://localhost:5000 in your browser

#### Option B: Command Line Interface
```bash
python futa_chatbot_simple.py
```

## Web Interface Features

- 🎨 **Modern Design**: Beautiful gradient UI with smooth animations
- 💬 **Real-time Chat**: Instant messaging interface
- 📱 **Responsive**: Works perfectly on desktop, tablet, and mobile
- 💡 **Sample Questions**: Click to ask pre-written questions
- 🗑️ **Clear Chat**: Reset conversation with one click
- ⚡ **Loading States**: Visual feedback during processing
- 🔄 **Conversation Memory**: Maintains context across messages

## CLI Commands

- `help` - Show sample questions
- `clear` - Clear conversation history
- `history` - Show recent conversation
- `quit` / `exit` / `bye` - Exit the chatbot

## Sample Questions

- "What are the admission requirements for FUTA?"
- "What faculties does FUTA have?"
- "How can I contact FUTA?"
- "What research centers are available?"
- "What is the student population?"
- "What sports facilities are available?"
- "Tell me about the alumni network"
- "What international partnerships does FUTA have?"
- "What are the academic programs offered?"
- "What is the campus location and address?"

## Project Structure

```
danielfinalyear/
├── futa_knowledge_base.txt          # Source knowledge base
├── futa_embedding_creator_simple.py # Creates embeddings
├── futa_chatbot_simple.py          # CLI chatbot
├── app.py                          # Web server (Flask)
├── templates/
│   └── index.html                  # Web interface
├── futa_kb_embeddings.json         # Generated embeddings
├── requirements.txt                 # Python dependencies
├── .env                            # API key (create this)
└── README.md                       # This file
```

## How It Works

1. **Knowledge Processing**: The knowledge base is split into manageable chunks
2. **Embedding Creation**: Each chunk is converted to a vector using Gemini embeddings
3. **Semantic Search**: User queries are embedded and matched using cosine similarity
4. **Response Generation**: Relevant context is used to generate accurate responses
5. **Conversation Memory**: Previous interactions are maintained for context

## Technical Details

- **Embeddings**: Gemini embedding-001 model
- **Generation**: Gemini 2.5 Flash model
- **Similarity**: Cosine similarity with numpy
- **Storage**: JSON files (no database required)
- **Memory**: Last 10 conversation turns
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Requirements

- Python 3.7+
- Gemini API key
- Internet connection for API calls
- Modern web browser (for web interface)

## Troubleshooting

- **API Key Error**: Make sure your `.env` file contains the correct API key
- **Embeddings Not Found**: Run `futa_embedding_creator_simple.py` first
- **Import Errors**: Install dependencies with `pip install -r requirements.txt`
- **Web Server Issues**: Make sure port 5000 is available, or change it in `app.py`
- **Mobile Issues**: The web interface is responsive and should work on all devices
