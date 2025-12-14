def retrieve_context(db, question, top_k, threshold):
    results = db.similarity_search_with_score(question, k=top_k)

    relevant_docs = []
    for doc, score in results:
        if score < threshold:
            relevant_docs.append(doc)

    return relevant_docs