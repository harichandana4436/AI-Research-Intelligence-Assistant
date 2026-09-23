# 🤖 AI Research Intelligence Assistant

## 1. Problem Statement

Researchers and students often spend a lot of time reading research
papers, finding relevant information, understanding technical content,
summarizing papers, and comparing multiple papers.

They usually need to manually search through large PDF documents and use
different tools for document analysis and research support.

This project provides a **single AI-powered application** that helps
users upload research papers, ask questions, summarize papers, compare
papers, and retrieve relevant information using RAG.

------------------------------------------------------------------------

## 2. Main Objective of the Project

The main objective of this project is to develop an **AI-powered
Research Intelligence Assistant** that helps users understand and
analyze research papers efficiently.

The project helps users to:

-   Upload and process research papers in PDF format
-   Ask questions about uploaded research papers
-   Retrieve relevant information using hybrid search
-   Generate context-grounded AI answers
-   Summarize research papers
-   Compare multiple research papers
-   Maintain conversation memory
-   Extract paper metadata
-   Evaluate RAG responses

------------------------------------------------------------------------

## 3. Features of the Project

### 💬 Research Chat

Users can ask questions about uploaded research papers and receive
AI-generated answers based on relevant retrieved content.

### 📄 PDF Research Paper Processing

Users can upload one or more research papers in PDF format. The
application extracts and processes the text from the papers.

### 🔎 Hybrid Search

The application combines:

-   Vector similarity search
-   BM25 keyword search

This helps retrieve relevant content using both semantic meaning and
keyword matching.

### 🧠 RAG-Based Answers

Relevant document chunks are retrieved and provided as context to the
LLM before generating an answer.

### 📝 Paper Summarization

Users can select a research paper and generate an automatic summary.

### 📚 Paper Comparison

Users can select multiple research papers and generate a comparison
based on their content.

### 🧾 Metadata Extraction

The application extracts research-paper information such as:

-   Title
-   Authors
-   Year
-   DOI

### 🧠 Conversation Memory

The application maintains recent conversation history so users can ask
follow-up questions with contextual understanding.

### 📊 RAG Evaluation

The application provides diagnostic metrics including:

-   Context Relevance
-   Answer Grounding
-   Answer Relevance
-   Overall Score

------------------------------------------------------------------------

## 4. Workflow of the Project

1.  **User Input** -- The user interacts with the Research Intelligence
    Assistant through the Streamlit web interface. The user can upload
    research papers, ask questions, summarize papers, compare papers, or
    evaluate RAG responses.

2.  **PDF Processing** -- When a research paper is uploaded, the
    application extracts text from the PDF page by page.

3.  **Metadata Extraction** -- The application extracts available
    metadata such as title, authors, year, and DOI.

4.  **Text Chunking** -- The extracted document text is divided into
    smaller chunks so that relevant sections can be efficiently
    processed and retrieved.

5.  **Embedding Generation** -- The `all-MiniLM-L6-v2` Sentence
    Transformer converts document chunks into numerical embeddings.

6.  **Vector Storage** -- The generated embeddings and document metadata
    are stored in ChromaDB.

7.  **User Question** -- When the user asks a question, the application
    performs both vector similarity search and BM25 keyword search.

8.  **Hybrid Retrieval** -- The results from vector search and BM25
    search are combined using the configured retrieval weights.

9.  **Context Retrieval** -- The most relevant document chunks are
    selected as context for the user's question.

10. **Groq AI Processing** -- The retrieved context and user question
    are sent to the configured Groq LLM to generate a response.

11. **Response Display** -- The generated answer and retrieved source
    information are displayed in the Streamlit application.

12. **Additional Processing** -- The application can also summarize
    selected papers, compare multiple papers, maintain conversation
    history, and evaluate RAG responses.

------------------------------------------------------------------------

## Workflow Diagram

``` text
                    User
                     │
                     ▼
            Upload Research Paper
                     │
                     ▼
              PDF Text Extraction
                     │
                     ▼
             Metadata Extraction
                     │
                     ▼
               Text Chunking
                     │
                     ▼
            Generate Embeddings
                     │
                     ▼
                 ChromaDB
                     │
                     │
              User Asks Question
                     │
                     ▼
              Query Processing
                     │
              ┌──────┴──────┐
              ▼             ▼
        Vector Search     BM25 Search
              │             │
              └──────┬──────┘
                     ▼
              Hybrid Ranking
                     │
                     ▼
            Relevant Context
                     │
                     ▼
                 Groq LLM
                     │
                     ▼
             Generated Answer
                     │
                     ▼
           Answer + Sources
```

## 5. Tech Stack

  ---------------------------------------------------------------------
  Technology                         Purpose
  ---------------------------------- ----------------------------------
  **Python**                         Main programming language

  **Streamlit**                      Web application and user interface

  **Groq**                           LLM inference

  **GPT-OSS-120B**                   Configured language model

  **Sentence Transformers**          Generates text embeddings

  **all-MiniLM-L6-v2**               Embedding model

  **ChromaDB**                       Stores document embeddings and
                                     metadata

  **BM25**                           Keyword-based document retrieval

  **PyPDF**                          Extracts text from PDF research
                                     papers

  **python-dotenv**                  Loads environment variables

  **PyTorch**                        Machine-learning framework used by
                                     embedding components
  ---------------------------------------------------------------------

### AI Model

The project is configured to use:

``` text
openai/gpt-oss-120b
```

### Embedding Model

The project uses:

``` text
all-MiniLM-L6-v2
```

------------------------------------------------------------------------

## 6. Project Files

``` text
AI-Research-Intelligence-Assistant/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── chroma_db/
│
└── other project modules
```

### File Description

-   **`app.py`** → Main Streamlit application, research chat, paper
    library, summarization, comparison, and RAG evaluation interface.
-   **`requirements.txt`** → Contains the required Python packages.
-   **`.env`** → Stores API keys and project configuration.
-   **`.gitignore`** → Prevents sensitive and unnecessary files such as
    `.env` and `chroma_db/` from being uploaded to GitHub.
-   **`chroma_db/`** → Persistent ChromaDB storage used for document
    embeddings and metadata.

------------------------------------------------------------------------

## 7. How to Run

### Step 1: Clone the Project

``` bash
git clone <your-github-repository-url>
cd AI-Research-Intelligence-Assistant
```

### Step 2: Create a Virtual Environment

``` bash
python -m venv venv
```

Activate it:

**Windows**

``` bash
venv\Scripts\activate
```

**Mac/Linux**

``` bash
source venv/bin/activate
```

### Step 3: Install Required Packages

``` bash
pip install -r requirements.txt
```

### Step 4: Add Groq API Key

Create a `.env` file in the project folder:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

**Do not upload your real API key to GitHub.**

### Step 5: Run the Application

``` bash
streamlit run app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## 8. Future Improvements

The project can be improved in the future by adding:

-   Support for more document formats
-   Improved document reranking
-   Citation generation for AI answers
-   Human-labelled RAG evaluation benchmarks
-   User login and authentication
-   Multi-user research workspaces
-   Advanced paper comparison
-   Research-paper recommendation
-   Improved visualization and analytics

------------------------------------------------------------------------

## 9. Conclusion

The **AI Research Intelligence Assistant** is an AI-powered application
designed to make research-paper analysis easier and more efficient.

It brings **research chat, hybrid search, paper summarization, paper
comparison, conversation memory, metadata extraction, and RAG
evaluation** into one application.

By using **Python, Streamlit, ChromaDB, Sentence Transformers, BM25,
PyPDF, and Groq**, the project provides an interactive platform for
working with research papers through natural-language queries.

------------------------------------------------------------------------

## 👨‍💻 Author

**Your Name**

If you find this project useful, consider giving the repository a ⭐.
