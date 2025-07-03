# COEQWAL Equity Analysis Chatbot

An AI-powered chatbot application that analyzes documents through the lens of the COEQWAL Equity Framework, identifying alignment with recognitional, procedural, distributional, and structural dimensions of equity. The system provides insights on how documents align with or potentially misalign with four dimensions of equity.

## Features

- **Document Analysis**: Upload and analyze documents through the COEQWAL Equity Framework lens
- **Interactive Chat**: Ask specific questions about your document and get AI-powered responses
- **Multiple Document Formats**: Support for PDF, DOCX, TXT, and other common formats
- **Framework Integration**: Leverages a pre-built vector database of the COEQWAL Equity Framework
- **Privacy-Focused**: "End Chat" functionality to remove uploaded documents from OpenAI servers

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI**: OpenAI API with Retrieval-Augmented Generation (RAG)
- **Vector Database**: Local JSON-based database (`db_v9.json`)
- **Frontend**: HTML, CSS, JavaScript

## Prerequisites

- **Python 3.9+**: [python.org](https://www.python.org/downloads/)
- **Git**: [git-scm.com](https://git-scm.com/)
- **OpenAI API Key**: An active OpenAI account and API key
- **`db_v9.json` file**: Pre-built vector database containing the COEQWAL framework

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wong-ian/COEQWAL.git
cd coeqwal-chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY="sk-YOUR_ACTUAL_OPENAI_API_KEY_HERE"
```

### 5. Add the COEQWAL Framework Database

Ensure the `db_v9.json` file is placed in the root directory of the project.

## Running the Application

### 1. Start the FastAPI Server

With your virtual environment activated:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Access the Web Interface

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

## Usage Guide

1. **Upload a Document**: Click the upload button and select your document
2. **Start Chatting**: Ask questions about your document's alignment with equity principles
3. **Analyze Results**: Review the AI's analysis based on the COEQWAL Framework
4. **End Session**: Before closing, click "End Chat & Clean Up Resources" to delete your data from OpenAI servers


## Privacy Considerations

- Documents uploaded to this application are temporarily stored on OpenAI servers
- Always use the "End Chat & Clean Up Resources" button to delete your data
