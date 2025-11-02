# import os
# import pickle
# import numpy as np
# import faiss
# from ollama_client import get_embedding, generate_answer

# STORAGE_DIR = "storage"
# INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
# META_PATH = os.path.join(STORAGE_DIR, "meta.pkl")

# def normalize(vecs: np.ndarray) -> np.ndarray:
#     return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

# def load_index():
#     """Load FAISS index + metadata from disk."""
#     if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
#         raise FileNotFoundError("Index not found. Run rag_ingest.py first.")
    
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         metadata = pickle.load(f)
#     return index, metadata

# def search_similar_chunks(query: str, index, metadata, top_k: int = 5):
#     """Return top_k most relevant chunks for the query (with scores)."""
#     query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
#     query_vec = normalize(query_vec)

#     scores, indices = index.search(query_vec, top_k)

#     results = []
#     for score, idx in zip(scores[0], indices[0]):
#         if idx < len(metadata):
#             results.append({
#                 "chunk": metadata[idx],
#                 "score": float(score)
#             })
#     return results

# def build_context(similar_chunks):
#     """Format retrieved chunks into a context block for the LLM."""
#     parts = []
#     for i, result in enumerate(similar_chunks):
#         ch = result["chunk"]
#         source = ch.get("source", "unknown")
#         page = ch.get("page", "unknown")
#         text = ch.get("text", "")
#         parts.append(
#             f"[Doc:{os.path.basename(source)} | Page {page}]\n{text}"
#         )
#     return "\n\n".join(parts)

# def make_answer_prompt(context_block: str, question: str) -> (str, str):
#     """
#     Returns (system_prompt, user_prompt) with nicer instructions for style.
#     """
#     system_prompt = (
#         "You are a retrieval-augmented assistant. "
#         "You answer questions using ONLY the provided context. "
#         "If the context is not enough to fully define or explain, you must say so. "
#         "Do not hallucinate outside the context. "
#         "Write clearly for an intelligent non-expert. Use bullet points if helpful."
#     )

#     user_prompt = f"""You are helping a user understand information from documents.

# Context from the documents:
# {context_block}

# User question:
# {question}

# Instructions for your answer:
# 1. Start with the clearest possible direct answer or definition if the context supports it.
# 2. Then give 2-4 short bullet points with supporting details or implications from the context.
# 3. If the context doesn't fully answer, explicitly say what is missing.
# 4. Do NOT mention 'I don't have internet' or 'the model'. Just focus on what the documents say.
# 5. Be concise, not legalistic.

# Now write the answer.
# """

#     return system_prompt, user_prompt

# def dedupe_sources(similar_chunks):
#     """
#     For printing sources at the end in a nice way:
#     collapse duplicate (source,page) pairs.
#     """
#     seen = set()
#     cleaned = []
#     for r in similar_chunks:
#         ch = r["chunk"]
#         src = ch.get("source", "unknown")
#         page = ch.get("page", "unknown")
#         key = (src, page)
#         if key not in seen:
#             seen.add(key)
#             cleaned.append({
#                 "source": src,
#                 "page": page,
#                 "score": r["score"],
#                 "preview": ch.get("text", "")[:200].replace("\n", " ").strip()
#             })
#     return cleaned

# def run_single_query(question: str, index, metadata, top_k: int = 5):
#     """
#     1. Retrieve relevant chunks
#     2. Build prompt
#     3. Ask llama3
#     4. Return final answer text and cleaned source info
#     """
#     similar_chunks = search_similar_chunks(question, index, metadata, top_k=top_k)
#     context_block = build_context(similar_chunks)
#     system_prompt, user_prompt = make_answer_prompt(context_block, question)
#     answer_text = generate_answer(system_prompt, user_prompt)
#     sources_clean = dedupe_sources(similar_chunks)
#     return answer_text, sources_clean

# if __name__ == "__main__":
#     # interactive loop
#     print("RAG chat. Ask anything about your loaded PDFs.")
#     print("Type 'exit' or 'quit' to stop.\n")

#     index, metadata = load_index()

#     while True:
#         question = input("You: ").strip()
#         if question.lower() in ["exit", "quit"]:
#             print("bye 👋")
#             break

#         answer_text, sources_used = run_single_query(question, index, metadata, top_k=5)

#         print("\nAssistant:\n")
#         print(answer_text)

#         print("\nSources referenced:")
#         for s in sources_used:
#             print(f"- {os.path.basename(s['source'])}, page {s['page']} (score {round(s['score'],4)})")
#         print("\n" + "-"*60 + "\n")

import os
import pickle
import numpy as np
import faiss
from gemini_client import get_embedding, generate_answer

STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.bin")
META_PATH = os.path.join(STORAGE_DIR, "meta.pkl")

# ---------------------------
# Helper utilities
# ---------------------------

def normalize(vecs: np.ndarray) -> np.ndarray:
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

def load_index():
    """Load FAISS index + metadata from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index not found. Run rag_ingest.py first.")
    
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_similar_chunks(query: str, index, metadata, top_k: int = 5):
    """Return top_k most relevant chunks for the query (with scores)."""
    query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    query_vec = normalize(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            results.append({
                "chunk": metadata[idx],
                "score": float(score)
            })
    return results

def build_context(similar_chunks):
    """
    Format retrieved chunks into a context block for the LLM.
    We keep doc + page info, but we won't force the model
    to repeat 'Source' every time in the answer.
    """
    parts = []
    for i, result in enumerate(similar_chunks):
        ch = result["chunk"]
        source = os.path.basename(ch.get("source", "unknown"))
        page = ch.get("page", "unknown")
        text = ch.get("text", "")
        parts.append(
            f"[{source} | page {page}]\n{text}"
        )
    return "\n\n".join(parts)

def dedupe_sources(similar_chunks):
    """Return unique (source,page) pairs with a short preview."""
    seen = set()
    cleaned = []
    for r in similar_chunks:
        ch = r["chunk"]
        src = ch.get("source", "unknown")
        page = ch.get("page", "unknown")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            cleaned.append({
                "source": os.path.basename(src),
                "page": page,
                "score": r["score"],
                "preview": ch.get("text", "")[:180].replace("\n", " ").strip()
            })
    return cleaned

def make_answer_prompt(context_block: str, question: str):
    """
    Build the (system_prompt, user_prompt) we send to the LLM.
    We emphasize readability:
    - Give a short summary first
    - Then bullet list of key points
    - If missing info, say clearly
    """
    system_prompt = (
        "You are a retrieval-augmented assistant. "
        "Your job is to summarize clearly what the provided documents say. "
        "You MUST follow these rules:\n"
        "1. Only use information from the provided context.\n"
        "2. If the context doesn't answer, say that directly.\n"
        "3. Write for a smart non-expert.\n"
        "4. Format the response in two parts:\n"
        "   - A short plain-language answer paragraph (2-4 sentences)\n"
        "   - Then a section called 'Key points:' with bullet points\n"
        "5. Do NOT mention 'Source 1', 'Source 2', etc in the main answer.\n"
        "6. Do NOT talk about 'the model' or 'I don't have internet'.\n"
    )

    user_prompt = f"""Context from documents:
{context_block}

User question:
{question}

Now write the answer in this exact shape:

Short answer:
[write a clean, user-friendly summary]

Key points:
- [bullet 1]
- [bullet 2]
- [bullet 3]
- [add more bullets if useful]

If the context doesn't fully answer the question, make that clear in the Short answer.
"""

    return system_prompt, user_prompt

def run_single_query(question: str, index, metadata, top_k: int = 5):
    """
    1. Retrieve relevant chunks
    2. Build prompt
    3. Ask llama3
    4. Return final answer text and source info
    """
    similar_chunks = search_similar_chunks(question, index, metadata, top_k=top_k)
    context_block = build_context(similar_chunks)
    system_prompt, user_prompt = make_answer_prompt(context_block, question)
    answer_text = generate_answer(system_prompt, user_prompt)
    sources_used = dedupe_sources(similar_chunks)
    return answer_text, sources_used


# ---------------------------
# CLI loop
# ---------------------------

if __name__ == "__main__":
    print("RAG chat. Ask anything about your loaded PDFs.")
    print("Type 'exit' or 'quit' to stop.\n")

    index, metadata = load_index()

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("bye 👋")
            break

        answer_text, sources_used = run_single_query(question, index, metadata, top_k=5)

        # Pretty print
        print("\n================================ ANSWER ================================\n")
        print(answer_text.strip())

        print("\n=============================== SOURCES ===============================\n")
        for s in sources_used:
            print(f"- {s['source']} | page {s['page']} | score {round(s['score'],4)}")
            print(f"  ↳ {s['preview']}")
            print()
        print("========================================================================\n")