#NO LLM INVOLVED IN THIS PROCESS PURELY OBTAINED RESULT FROM EMBEDDINGS ACCORDIGLY...


from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ FIX
#loading up environment variables...
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #same model used in ingestion_pipeline.

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

query = "which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k":5})#Top 3 outcomes from files retrieved out...
relevant_docs = retriever.invoke(query)#invoking query into retriever

print(f"User Query: {query}")
print(" --- Context --- ")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")



