# scripts/ingest.py

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import shutil

DATA_DIR = "Dataset"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Nettoyage ancienne base
if os.path.exists(VECTORSTORE_DIR):
    shutil.rmtree(VECTORSTORE_DIR)

# Chargement des documents PDF
loaders = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loaders.append(PyPDFLoader(os.path.join(DATA_DIR, file)))

documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
texts = splitter.split_documents(documents)

# Embeddings biomédicaux (PubMedBERT)
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Indexation FAISS
db = FAISS.from_documents(texts, embeddings)
db.save_local(VECTORSTORE_DIR)

print(f"✅ Ingestion terminée. {len(texts)} chunks indexés.")
