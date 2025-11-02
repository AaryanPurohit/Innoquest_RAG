import os, pickle, numpy as np, faiss, time
from tqdm import tqdm
from chunking import build_chunks_from_pdf
from gemini_client import get_embedding

PDF_DIR = "data/pdfs"
STORAGE_DIR = "storage"
INDEX_PATH = f"{STORAGE_DIR}/faiss_index.bin"
META_PATH  = f"{STORAGE_DIR}/meta.pkl"

def normalize(vecs): return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

def ingest():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    pdfs = [f"{PDF_DIR}/{f}" for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    all_meta, all_vecs = [], []

    for pdf in pdfs:
        print(f"Processing {pdf}...")
        chunks = build_chunks_from_pdf(pdf)
        for ch in tqdm(chunks, desc="Embedding"):
            try:
                emb = np.array(get_embedding(ch["text"]), dtype="float32")
                all_vecs.append(emb)
                all_meta.append(ch)
                # Small delay to prevent overwhelming Ollama
                time.sleep(0.1)
            except Exception as e:
                print(f"Error embedding chunk: {e}")
                print(f"Chunk text preview: {ch['text'][:100]}...")
                continue

    all_vecs = normalize(np.vstack(all_vecs)).astype("float32")
    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f: pickle.dump(all_meta, f)

    print(f"\nSaved index ({len(all_meta)} chunks) ✅")

if __name__ == "__main__":
    ingest()