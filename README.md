# 📊 IntelliDecks – AI-Powered Project Discovery Platform

## 🚀 Overview

IntelliDecks is an AI-powered project portfolio assistant that enables users to explore, search, and analyze a repository of projects using natural language queries.

Instead of manually browsing hundreds of slides, users can simply ask questions such as:

* "Show ML/AI projects"
* "Which projects use AWS?"
* "Find telecom domain projects"
* "Compare NLP projects"
* "Show projects with team size 4"

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant project information from a PowerPoint-based project repository and generate intelligent responses.

---

## 🎯 Problem Statement

Organizations often maintain large repositories of project documentation in PowerPoint presentations. Searching through hundreds of slides to find relevant projects is time-consuming and inefficient.

IntelliDecks solves this problem by:

* Converting project slides into structured knowledge.
* Creating semantic embeddings.
* Storing project information in a vector database.
* Allowing users to query projects using natural language.

---

## ✨ Features

### 🔍 Natural Language Search

Ask project-related questions in plain English.

### 🤖 AI-Powered Responses

Uses Large Language Models (LLMs) to generate contextual answers.

### 📚 Semantic Retrieval

Retrieves relevant projects using vector similarity search.

### ⚡ Fast Search

Powered by FAISS for low-latency retrieval.

### 📊 Rich Project Metadata

Supports retrieval by:

* Project ID
* Project Title
* Domain
* Technology Stack
* Team Size
* Duration
* Project Type
* Expected Outcomes

### 🎨 Interactive Web Interface

Built with Streamlit for an intuitive user experience.

---

# 🏗 System Architecture

```text
                        ┌─────────────────────┐
                        │ Project PPT Dataset │
                        └──────────┬──────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                     │ PPT Parsing Engine      │
                     │ (python-pptx)          │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ Structured Project Data │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ Embedding Generation    │
                     │ MiniLM-L6-v2            │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ FAISS Vector Database   │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ Retrieval Layer         │
                     │ LangChain Retriever     │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ Groq LLM                │
                     │ Llama 3.1              │
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │ Streamlit Interface     │
                     └─────────────────────────┘
```

---

# ⚙️ Technology Stack

| Component              | Technology            |
| ---------------------- | --------------------- |
| Frontend               | Streamlit             |
| LLM                    | Groq (Llama 3.1)      |
| Framework              | LangChain             |
| Vector Database        | FAISS                 |
| Embeddings             | Sentence Transformers |
| PPT Processing         | python-pptx           |
| Environment Management | dotenv                |
| Language               | Python                |

---

# 📂 Project Structure

```text
INTELLIDECKS/
│
├── app.py
├── store_index.py
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   └── Dataset_project_repository.pptx
│
├── faiss_index/
│
└── src/
    └── ppt_parser.py
```

---

# 🔄 Workflow

## Step 1: Parse PPT

The system reads project information from PowerPoint slides.

Extracted fields:

* Project ID
* Project Title
* Domain
* Team Size
* Duration
* Project Type
* Technology Stack

---

## Step 2: Create Documents

Each project is converted into a LangChain Document.

```python
Document(
    page_content=project_text,
    metadata=project_metadata
)
```

---

## Step 3: Generate Embeddings

Project descriptions are transformed into vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## Step 4: Store in FAISS

Embeddings are stored inside a FAISS vector database for fast semantic retrieval.

---

## Step 5: Query Processing

User asks:

```text
Show ML/AI projects using AWS
```

The system:

1. Converts query to embeddings.
2. Searches FAISS.
3. Retrieves relevant projects.
4. Sends context to LLM.
5. Generates final answer.

---

# 🧠 Retrieval-Augmented Generation (RAG)

IntelliDecks follows the RAG architecture:

```text
User Query
      │
      ▼
Embedding Model
      │
      ▼
FAISS Retrieval
      │
      ▼
Relevant Projects
      │
      ▼
Groq LLM
      │
      ▼
Final Response
```

This ensures answers remain grounded in the project repository.

---

# ▶️ Installation

## Clone Repository

```bash
git clone <repository-url>
cd IntelliDecks
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 📦 Build Vector Database

Run:

```bash
python store_index.py
```

Expected Output:

```text
Extracted 100 projects
Created 100 document chunks
FAISS index saved to 'faiss_index/'
```

---

# 🚀 Launch Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 📸 Sample Queries

```text
Show all ML/AI projects

Which projects use AWS?

Find telecom projects

Show projects with team size 4

Compare NLP projects

Which project uses XGBoost?

Show projects with measurable business impact
```

---

# 🎯 Use Cases

* Enterprise Project Discovery
* Portfolio Management
* Knowledge Retrieval
* Project Recommendation Systems
* Internal Documentation Search
* Project Analytics

---

# 🔮 Future Enhancements

* Multi-file PPT support
* Project similarity recommendations
* Analytics dashboard
* Domain-wise visualizations
* Role-based access control
* PDF and DOCX ingestion
* Project comparison view

---

# 👨‍💻 Developed By

Harsh Sharma

B.Tech (Artificial Intelligence)

Microsoft Certified Azure AI Engineer Associate

Built for AI/GenAI Hackathon

```
```
