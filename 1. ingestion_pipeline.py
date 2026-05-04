import os
#Read all kinds of files.
from langchain_community.document_loaders import TextLoader, DirectoryLoader
#After loading divide them into chunks.
from langchain_text_splitters import CharacterTextSplitter
#Converting to Embedding data.
from langchain_openai import OpenAIEmbeddings
#Chroma Database to host locally...
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path = "docs"):
    #Loads all kinds of files from directory...
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} don't exist. create and add accordingly.")
    
    loader = DirectoryLoader(path = docs_path , glob = "*.txt", loader_cls= TextLoader, loader_kwargs={"encoding": "utf-8"})  # FIX for windows...
    #read only txt files(mentioned glob here).

    documents = loader.load() #invoking the load option.

    if len(documents) == 0 :
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Add accordingly.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")  # FIX
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size = 800, chunk_overlap = 0):
    #split docs into smaller chunks with overlap.

    print("Splitting docs into Chunks")

    text_splitter = CharacterTextSplitter(chunk_size = chunk_size , chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents) #All the divided chunks places here accordingly...

    if chunks:
        for i, chunk in enumerate(chunks [:5]) :
            print(f"\n -- Chunk {i+1} --- ")
            print(f"Source: {chunk.metadata['source' ]}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n ... and {len(chunks) - 5} more chunks")

    return chunks


#gpt changed version... from gpt 3 small(paid) to huggingface(free)...
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    documents = load_documents("docs") #func1
    chunks = split_documents(documents) #func2
    vectorstore = create_vector_store(chunks) #func3
    print("\nAll documents loaded successfully!")

if __name__ == "__main__":
    main()
