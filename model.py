import pandas as pd
import shutil
import os
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. CLEANUP: Delete old database to ensure fresh ingestion
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("🗑️ Deleted old database.")

# 2. Load your NEW geocoded CSV
# UPGRADE: Pointing to the file that actually contains coordinates
df = pd.read_csv('ai_jobs_global_geocoded.csv')

# 3. Transform into Documents with Enhanced Metadata
documents = []
for _, row in df.iterrows():
    content = str(row['job_description']) if pd.notna(row['job_description']) else "No description provided."
    
    # UPGRADE: Build a comprehensive metadata dict for advanced filtering
    meta = {
        "source": str(row['company']) if pd.notna(row['company']) else "Unknown",
        "country": str(row['country']) if pd.notna(row['country']) else "Unknown",
        "city": str(row['city']) if pd.notna(row['city']) else "Unknown"
    }
    
    # Only inject lat/lon if they are valid numbers (Chroma prefers clean types)
    if pd.notna(row['lat']) and pd.notna(row['lon']):
        meta["lat"] = float(row['lat'])
        meta["lon"] = float(row['lon'])
    
    doc = Document(
        page_content=content, 
        metadata=meta
    )
    documents.append(doc)

# 4. Create new database
print("🧠 Generating embeddings and building ChromaDB (this might take a minute)...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents, 
    embedding_function, 
    persist_directory="./chroma_db"
)

print(f"🎉 Success! Ingested {len(documents)} documents with geo-metadata into ChromaDB.")