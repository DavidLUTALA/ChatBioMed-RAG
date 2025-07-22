# Scripts/pubmed_search.py

from pymed import PubMed

pubmed = PubMed(tool="ChatBioMed", email="your_email@example.com")  # ← remplace par ton vrai email

def search_pubmed(query: str, max_results: int = 5):
    """Effectue une recherche PubMed et retourne les résultats principaux."""
    results = pubmed.query(query, max_results=max_results)

    publications = []
    for article in results:
        title = article.title
        abstract = article.abstract
        authors = ", ".join([a['lastname'] for a in article.authors if a.get('lastname')])
        journal = article.journal
        pub_date = article.publication_date
        doi = article.doi

        publications.append({
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "date": pub_date,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else None
        })

    return publications
