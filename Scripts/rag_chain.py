# scripts/rag_chain.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- LLM : Mistral via Ollama ---
llm = OllamaLLM(model="mistral")

# --- Chargement des embeddings biomédicaux ---
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# --- Chargement base vectorielle FAISS ---
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# --- Prompt de RAG (conservatif) ---
prompt_template = """
You are a biomedical expert. Use the following context to answer the question.
If the answer is not in the context, say you don’t know. Don't invent.

Context: {context}

Question: {question}

Helpful Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# --- Test simple ---
if __name__ == "__main__":
    question = input("🧠 Ask a biomedical question: ")
    result = qa_chain(question)
    
    print("\n🔎 Answer:\n", result["result"])
    print("\n📄 Sources used:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']}")
