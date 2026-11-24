#-------------------------------------------------------------------------
# AUTHOR: Harman Sekhon
# FILENAME: vsm_search.py
# SPECIFICATION: Ranks documents using TF-IDF and cosine similarity, then
#                evaluates each query using AP and the search engine using MAP.
# FOR: CS 4250 - Assignment #4
# TIME SPENT: 1 hr 30 min
#-------------------------------------------------------------------------

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
docs = pd.read_csv("docs.csv")
queries = pd.read_csv("queries.csv")
relevance = pd.read_csv("relevance_judgments.csv")

# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(docs["text"].fillna(""))

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
ap_scores = []

for _, query in queries.iterrows():
    query_id = query["query_id"]
    query_vector = vectorizer.transform([str(query["query_text"])])

    similarities = cosine_similarity(query_vector, document_vectors).ravel()
    ranked_indices = sorted(range(len(docs)), key=lambda i: (-similarities[i], i))

    judgments = relevance[relevance["query_id"] == query_id]
    relevant_ids = set(judgments.loc[judgments["judgment"] == "R", "doc_id"])

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    found_relevant = 0
    precision_sum = 0.0
    for rank, doc_index in enumerate(ranked_indices, start=1):
        doc_id = docs.iloc[doc_index]["doc_id"]
        if doc_id in relevant_ids:
            found_relevant += 1
            precision_sum += found_relevant / rank

    ap = precision_sum / len(relevant_ids) if relevant_ids else 0.0

    # store the AP value for this query
    ap_scores.append((query_id, ap))

# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
ap_scores.sort(key=lambda item: (-item[1], item[0]))
map_score = sum(ap for _, ap in ap_scores) / len(ap_scores) if ap_scores else 0.0

# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
for query_id, ap in ap_scores:
    print(f"{query_id}: {ap:.6f}")
print(f"Mean Average Precision (MAP): {map_score:.6f}")
