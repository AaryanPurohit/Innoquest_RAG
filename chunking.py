import re
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader

############################
# 1. PDF text extraction
############################

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, cleaned_page_text)
    page_number is 1-based.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_whitespace(raw)
        pages.append((i + 1, cleaned))
    return pages

def clean_whitespace(text: str) -> str:
    # normalize all whitespace/newlines to single spaces
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


############################
# 2. Paragraph splitting
############################

def split_into_paragraphs(page_text: str) -> List[str]:
    """
    Try to detect paragraph-ish blocks.
    We'll split on blank lines OR large gaps, then normalize.
    """
    # First bring back some structure before cleaning:
    # We'll consider double newlines a paragraph break.
    # If PDF extraction already collapsed newlines, we won't hurt anything.
    rough_paras = re.split(r"\n\s*\n+", page_text)

    cleaned_paras = []
    for p in rough_paras:
        p = p.replace("\x00", " ")
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned_paras.append(p)
    return cleaned_paras


############################
# 3. Sliding window fallback
############################

def sliding_windows(text: str,
                    chunk_size: int,
                    overlap: int) -> List[str]:
    """
    Pure character-based fallback.
    """
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        out.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return out


############################
# 4. Smart chunker
############################

def smart_chunks(page_text: str,
                 chunk_size: int = 800,
                 overlap: int = 200) -> List[str]:
    """
    Strategy:
    - Split into paragraphs.
    - Greedily pack paragraphs into ~chunk_size.
    - If a single paragraph is huge, break it with sliding window.
    - After we get base chunks, add overlap between consecutive chunks
      by prepending tail of previous chunk.
    """
    paragraphs = split_into_paragraphs(page_text)

    base_chunks = []
    buffer = ""

    for para in paragraphs:
        # Case 1: paragraph fits if we append it
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + " " + para).strip()

        else:
            # Flush current buffer first (if not empty)
            if buffer:
                base_chunks.append(buffer)
                buffer = ""

            # Case 2: paragraph itself is longer than chunk_size
            if len(para) > chunk_size:
                long_parts = sliding_windows(para, chunk_size, overlap)
                base_chunks.extend(long_parts)
            else:
                buffer = para

    # Flush remainder
    if buffer:
        base_chunks.append(buffer)

    # Now create overlap across chunks:
    # Take the last `overlap` chars of previous chunk and prepend to next.
    final_chunks = []
    for i, ch in enumerate(base_chunks):
        if i == 0:
            final_chunks.append(ch)
        else:
            prev = final_chunks[-1]
            tail = prev[-overlap:]
            merged = (tail + " " + ch).strip()
            final_chunks.append(merged)

    return final_chunks


############################
# 5. High-level builder
############################

def build_chunks_from_pdf(pdf_path: str,
                          chunk_size: int = 800,
                          overlap: int = 200) -> List[Dict]:
    """
    Returns list of dicts:
    {
        "text": chunk_text,
        "source": pdf_path,
        "page": page_number,
        "chunk_id": f"{pdf_path}_p{page}_c{k}"
    }

    We apply smart_chunks() page-by-page so we still keep page metadata.
    """
    page_texts = extract_text_from_pdf(pdf_path)
    all_chunks: List[Dict] = []

    for page_num, page_txt in page_texts:
        if not page_txt.strip():
            continue

        page_chunks = smart_chunks(
            page_txt,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for k, ch in enumerate(page_chunks):
            all_chunks.append({
                "text": ch,
                "source": pdf_path,
                "page": page_num,
                "chunk_id": f"{pdf_path}_p{page_num}_c{k}"
            })

    return all_chunks