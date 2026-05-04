# NO LLM API (FREE VERSION) → Using HuggingFace instead of OpenAI

import os

# Chroma DB for vector storage and retrieval
from langchain_chroma import Chroma

# HuggingFace embeddings (same model used in ingestion pipeline)
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace LLM pipeline
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Loading environment variables (good practice, even if not used here)
from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# Step 1: Load Existing Vector DB
# -----------------------------
persistent_directory = "db/chroma_db"

# IMPORTANT: Same embedding model must be used as ingestion phase
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load ChromaDB (No chunks needed here → already stored)
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)


# -----------------------------
# Step 2: Define User Query
# -----------------------------
query = "which island does SpaceX lease for its launches in the Pacific?"

# Retriever fetches top-k relevant chunks using cosine similarity (default)
retriever = db.as_retriever(search_kwargs={"k": 5})  # Top 5 results retrieved

# Invoke retriever with query → returns relevant document chunks
relevant_docs = retriever.invoke(query)


# -----------------------------
# Step 3: Combine Retrieved Context
# -----------------------------
# Combine all retrieved chunks into a single context string
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Prompt template → guides the LLM behavior
combined_input = f"""
Answer ONLY using the provided documents.
If the answer is not found, say "Not found in context".

Documents:
{context}

Question:
{query}

Answer:
"""


# -----------------------------
# Step 4: HuggingFace LLM (FLAN-T5)
# -----------------------------
# Using text2text-generation → correct task for FLAN-T5
hf_pipeline = pipeline(
    "text-generation",   # ✅ FIXED (compatible) also lightweight...
    model="google/flan-t5-base",
    max_length=512
)

# Wrap HuggingFace pipeline with LangChain interface
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# -----------------------------
# Step 5: Generate Final Answer
# -----------------------------
# Pass combined prompt to LLM
response = llm.invoke(combined_input)

print("\nUser Query:")
print(query)

print("\n--- Retrieved Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}")

print("\n--- Generated Response ---")
print(response)
