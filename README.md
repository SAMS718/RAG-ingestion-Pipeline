# 🧠 RAG Ingestion Pipeline (LangChain + Chroma + HuggingFace)

This project implements a **Retrieval-Augmented Generation (RAG) ingestion pipeline** using LangChain. It processes raw text documents, converts them into embeddings, and stores them in a vector database for efficient semantic search and retrieval.

---

## 🚀 Features

* 📂 Load `.txt` documents from a directory
* ✂️ Split documents into manageable chunks
* 🔎 Generate embeddings using HuggingFace (free & offline)
* 🗄️ Store embeddings in Chroma vector database
* ⚡ Ready for integration with a RAG-based chatbot

---

## 🏗️ Pipeline Architecture

```
Documents → Loader → Text Splitter → Embeddings → Vector Store (Chroma)
```

---

## 📁 Project Structure

```
RAG_Code/
│
├── docs/                # Input text files
├── db/
│   └── chroma_db/       # Persisted vector database
├── ingestion_pipeline.py
├── .env
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-ingestion-pipeline.git
cd rag-ingestion-pipeline
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

#### Activate environment:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-text-splitters sentence-transformers python-dotenv
```

---

## 📄 Usage

### 1️⃣ Add documents

Place `.txt` files inside the `docs/` folder.

### 2️⃣ Run the pipeline

```bash
python ingestion_pipeline.py
```

---

## 🔍 What Happens Internally

### 📥 Step 1: Document Loading

* Loads `.txt` files from `docs/` directory
* Validates file existence

### ✂️ Step 2: Text Splitting

* Splits documents into chunks (default: 800 characters)
* Helps improve embedding quality

### 🧠 Step 3: Embedding Generation

* Uses **HuggingFace model: `all-MiniLM-L6-v2`**
* Completely free and runs locally

### 🗄️ Step 4: Vector Storage

* Stores embeddings in **ChromaDB**
* Persists data in `db/chroma_db/`

---

## 🧪 Sample Output

```
Document 1:
Source: docs/sample.txt
Content length: 1200 characters

-- Chunk 1 ---
Length: 800 characters

Vector store created and saved to db/chroma_db
```

---

## 🔥 Key Highlights

* ✅ No OpenAI API required (cost-free setup)
* ✅ Works offline after first model download
* ✅ Modular pipeline (easy to extend for full RAG system)
* ✅ Beginner-friendly and interview-ready project

---

## ⚠️ Notes

* Ensure `docs/` folder contains `.txt` files
* First run may download embedding model (~100MB)
* For large datasets, consider chunk overlap tuning

---

## 🚀 Future Improvements

* Add retriever and query system
* Integrate LLM for full RAG chatbot
* Support PDF/HTML document loaders
* Add UI (Streamlit / React)

---

## 🤝 Contribution

Feel free to fork this repo and enhance it with:

* Better chunking strategies
* Advanced embedding models
* Hybrid search

---

## 📜 License

This project is open-source and available under the MIT License.
