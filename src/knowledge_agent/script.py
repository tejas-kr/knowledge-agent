
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import fitz
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "documents"

EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
GENERATION_MODEL = "qwen3:1.7b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

# Ollama runs locally by default at http://localhost:11434
client = ollama.Client(host="http://localhost:11434")


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def load_pdf(path: Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF while preserving page metadata.
    """

    documents = []

    with fitz.open(path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append({
                "document_id": path.stem,
                "filename": path.name,
                "page_number": page_number,
                "content": text,
            })

    return documents


def load_all_pdfs() -> list[dict[str, Any]]:
    """
    Discover and load all PDFs from documents/.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    print(f"\nFound {len(pdf_files)} PDF files.")

    all_pages = []

    for pdf_path in pdf_files:

        print(f"Loading: {pdf_path.name}")

        try:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)

            print(f"  Extracted {len(pages)} pages.")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nTotal pages extracted: {len(all_pages)}")

    return all_pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be >= 0 and < chunk_size")

    words = text.split()

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def create_chunks(
    pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert extracted pages into chunks with source metadata.
    """

    chunks = []

    for page in pages:

        page_chunks = chunk_text(page["content"])

        for chunk_index, content in enumerate(page_chunks):

            chunks.append({
                "chunk_id": (
                    f'{page["document_id"]}_'
                    f'p{page["page_number"]}_'
                    f'c{chunk_index}'
                ),
                "document_id": page["document_id"],
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "content": content,
            })

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Generate embeddings using the local Ollama embedding model.

    Processes inputs in batches to avoid excessively large requests.
    """

    embeddings = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        response = client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
            truncate=True,
        )

        embeddings.extend(response["embeddings"])

        completed = min(start + len(batch), len(texts))

        print(
            f"\rEmbedding: {completed}/{len(texts)}",
            end="",
            flush=True,
        )

    print()

    return embeddings


# ============================================================
# VECTOR SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def build_knowledge_base() -> list[dict[str, Any]]:
    """
    Load PDFs, chunk content, and generate embeddings.
    """

    pages = load_all_pdfs()

    if not pages:
        raise RuntimeError("No extractable PDF text found.")

    chunks = create_chunks(pages)

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    print("\nKnowledge base successfully initialized.")

    return chunks


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question: str,
    knowledge_base: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve the most semantically similar chunks.
    """

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=question,
        truncate=True,
    )

    query_embedding = response["embeddings"][0]

    scored_chunks = []

    for chunk in knowledge_base:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        scored_chunks.append({
            **chunk,
            "score": score,
        })

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


# ============================================================
# RAG GENERATION
# ============================================================

SYSTEM_PROMPT = """
You are a technical research assistant.

Answer the user's question using the supplied context.

Rules:
1. Ground your answer in the supplied context.
2. Do not invent facts or citations.
3. If the context is insufficient, explicitly say so.
4. Explain technical concepts clearly.
5. Cite supporting sources using the source labels provided.
6. Do not claim that a source supports something unless
   the retrieved passage actually supports it.

The context contains excerpts from research papers.
"""


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Generate an answer using Qwen through Ollama.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        source_label = (
            f"[S{index}: {chunk['filename']}, "
            f"page {chunk['page_number']}]"
        )

        context_parts.append(
            f"{source_label}\n{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}

Provide a detailed answer based on the context.
Include source labels such as [S1] in your answer.
"""

    response = client.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    )

    return response["message"]["content"]


# ============================================================
# CHAT INTERFACE
# ============================================================

def ask_question(
    question: str,
    knowledge_base: list[dict[str, Any]],
) -> None:
    """
    Retrieve context and generate an answer.
    """

    print("\nSearching knowledge base...")

    retrieved_chunks = retrieve(
        question,
        knowledge_base,
    )

    print("\nRetrieved sources:")

    for index, chunk in enumerate(retrieved_chunks, start=1):

        print(
            f"  [S{index}] {chunk['filename']} "
            f"| Page {chunk['page_number']} "
            f"| Similarity: {chunk['score']:.4f}"
        )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PERSONAL KNOWLEDGE BASE AGENT")
    print("=" * 70)

    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Generation model: {GENERATION_MODEL}")

    try:
        knowledge_base = build_knowledge_base()

    except Exception as exc:
        print(f"\nInitialization failed: {exc}")
        sys.exit(1)

    print("\nKnowledge base ready.")
    print("Ask questions about your research papers.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            ask_question(
                question,
                knowledge_base,
            )

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
