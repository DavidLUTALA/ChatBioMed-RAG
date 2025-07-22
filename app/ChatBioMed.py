#Import Bibliothèques
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from Scripts.evaluation import evaluate_similarity
from Scripts.pubmed_search import search_pubmed

# LLM local
llm = OllamaLLM(model="mistral")

# Charger FAISS et embeddings
from Scripts.pubmedbert_embedding import PubMedBERTEmbedder
embedder = PubMedBERTEmbedder()
class CustomEmbedWrapper:
    def embed_documents(self, texts): return embedder.embed_batch(texts).tolist()
    def embed_query(self, text): return embedder.embed(text).tolist()
    def __call__(self, text): return self.embed_query(text)
embeddings = CustomEmbedWrapper()
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Prompt
prompt_template = """
You are a biomedical expert. Use the following context to answer the question.
If the answer is not in the context, say you don’t know. Don't invent.

Context: {context}

Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Streamlit config
st.set_page_config(page_title="🧬 Biomedical QA", layout="wide")

# Session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "active_question_index" not in st.session_state:
    st.session_state.active_question_index = None

# CSS
st.markdown("""
    <style>
    .block-container { padding: 2rem 4rem; }
    .sidebar-history button {
        width: 100%;
        margin-bottom: 0.4rem;
        text-align: left;
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
    }
    .source-box {
        background-color: #f1f3f5;
        border-left: 4px solid #339af0;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Layout principal
col_main, col_history = st.columns([3, 1], gap="large")

# Historique à droite
with col_history:
    st.markdown("### Historique")
    if not st.session_state.qa_history:
        st.caption("Aucune question posée.")
    else:
        for idx, item in enumerate(st.session_state.qa_history):
            if st.button(f" {item['question'][:60]}", key=f"q_{idx}"):
                st.session_state.active_question_index = idx

# Colonne principale
with col_main:
    st.markdown("# Biomedical-RAG")
    st.markdown("Pose une question médicale basée sur les documents indexés (Cas du cancer).")

    question = st.text_input(" Votre question", placeholder="e.g. What is the treatment for glioblastoma multiforme?")

    if question and st.button(" Poser votre question"):
        with st.spinner("Analyse en cours..."):
            result = qa_chain(question)
            answer = result["result"]
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content
                }
                for doc in result["source_documents"]
            ]
            st.session_state.qa_history.insert(0, {
                "question": question,
                "answer": answer,
                "sources": sources
            })
            st.session_state.active_question_index = 0

    # Affichage réponse sélectionnée
    if st.session_state.active_question_index is not None:
        item = st.session_state.qa_history[st.session_state.active_question_index]
        st.markdown("##  Réponse")
        st.write(item["answer"])

        #st.markdown("## Évaluation automatique")
        #score = evaluate_similarity(item["answer"], [src["content"] for src in item["sources"]])
        #st.metric(" Similarité réponse/sources", f"{score * 100:.2f} %")

        st.markdown("## Sources utilisées")
        for i, src in enumerate(item["sources"]):
            st.markdown(f"**🔹 Source {i+1} – `{src['source']}`**")
            st.markdown(f"<div class='source-box'>{src['content']}</div>", unsafe_allow_html=False)

# Recherche PubMed
st.markdown("## Recherche en ligne des articles sur PubMed")
pubmed_query = st.text_input("Rechercher un sujet sur PubMed", placeholder="e.g. metastatic prostate cancer")

if pubmed_query and st.button(" Rechercher dans PubMed"):
    with st.spinner("Recherche dans PubMed..."):
        results = search_pubmed(pubmed_query, max_results=5)
        if not results:
            st.warning("Aucun résultat trouvé.")
        else:
            st.success(f"{len(results)} résultats trouvés.")
            for i, res in enumerate(results):
                st.markdown(f"### Résultat {i+1}")
                st.markdown(f"**Titre :** {res['title']}")
                st.markdown(f"**Auteurs :** {res['authors']}")
                st.markdown(f"**Journal :** {res['journal']} ({res['date']})")
                if res['url']:
                    st.markdown(f"[🔗 DOI]({res['url']})", unsafe_allow_html=True)
                if res['abstract']:
                    with st.expander("📘 Voir l’abstract"):
                        st.write(res['abstract'])
