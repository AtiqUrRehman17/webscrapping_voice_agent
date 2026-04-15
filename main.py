import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────
# 0. Load environment
# ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file.")

CHROMA_DIR = "./chroma_db"          # where ChromaDB persists data
COLLECTION  = "website_rag"         # collection name inside ChromaDB


# ──────────────────────────────────────────────
# 1. WEB SCRAPER
# ──────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def is_same_domain(base_url: str, link: str) -> bool:
    """Return True if link belongs to the same domain as base_url."""
    base_netloc = urlparse(base_url).netloc
    link_netloc = urlparse(link).netloc
    return link_netloc == "" or link_netloc == base_netloc


def clean_text(text: str) -> str:
    """Remove excessive whitespace and blank lines."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_page(url: str) -> tuple[str, list[str], list[str]]:
    """
    Scrape a single page.
    Returns:
        text       – cleaned visible text
        links      – list of absolute internal hrefs found on the page
        image_info – list of strings describing images (alt text or src)
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return "", [], []

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Remove noise tags ──
    for tag in soup(["script", "style", "noscript", "head",
                     "nav", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    # ── Extract text ──
    text = clean_text(soup.get_text(separator=" "))

    # ── Extract internal links ──
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip anchors, mailto, tel, javascript
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_href = urljoin(url, href)
        # Strip fragment
        abs_href = abs_href.split("#")[0]
        if is_same_domain(url, abs_href):
            links.append(abs_href)

    # ── Extract image info ──
    image_info = []
    for img in soup.find_all("img"):
        alt  = img.get("alt", "").strip()
        src  = img.get("src", "").strip()
        if alt:
            image_info.append(f"[IMAGE: {alt}]")
        elif src:
            image_info.append(f"[IMAGE src: {src}]")

    return text, list(set(links)), image_info


def crawl_website(start_url: str, max_pages: int = 50) -> list[Document]:
    """
    BFS crawl starting from start_url.
    Returns a list of LangChain Document objects (one per page).
    """
    print(f"\n{'='*60}")
    print(f"Starting crawl: {start_url}")
    print(f"Max pages     : {max_pages}")
    print("="*60)

    visited = set()
    queue   = deque([start_url])
    docs    = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        print(f"  Scraping [{len(visited)+1}/{max_pages}]: {url}")
        visited.add(url)

        text, links, images = scrape_page(url)

        if not text:
            continue

        # Append image descriptions to the page text
        if images:
            image_block = "\n".join(images)
            text = text + "\n\n" + image_block

        # Wrap in a LangChain Document
        docs.append(Document(
            page_content=text,
            metadata={"source": url}
        ))

        # Enqueue new links
        for link in links:
            if link not in visited:
                queue.append(link)

        time.sleep(0.5)   # polite delay

    print(f"\nCrawl complete. Pages scraped: {len(docs)}")
    return docs


# ──────────────────────────────────────────────
# 2. CHUNKING
# ──────────────────────────────────────────────

def chunk_documents(docs: list[Document],
                    chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list[Document]:
    """Split each document into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunking complete. Total chunks: {len(chunks)}")
    return chunks


# ──────────────────────────────────────────────
# 3. EMBEDDINGS + CHROMADB
# ──────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load a local open-source embedding model.
    'all-MiniLM-L6-v2' is fast and high quality (~80 MB).
    Swap for 'BAAI/bge-small-en-v1.5' or 'sentence-transformers/all-mpnet-base-v2'
    if you want higher accuracy.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nLoading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded.")
    return embeddings


def build_vector_store(chunks: list[Document],
                       embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Embed all chunks and persist them into ChromaDB."""
    print(f"\nBuilding ChromaDB vector store at '{CHROMA_DIR}' ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print("Vector store built and persisted.")
    return vectordb


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load an existing ChromaDB store from disk."""
    print(f"\nLoading existing ChromaDB from '{CHROMA_DIR}' ...")
    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vector store loaded.")
    return vectordb


# ──────────────────────────────────────────────
# 4. RAG RETRIEVAL + QA
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def build_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Build the RAG QA chain using Gemini as the LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}   # retrieve top-5 most relevant chunks
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # stuff all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return qa_chain


def interactive_qa(qa_chain: RetrievalQA):
    """Simple interactive loop for asking questions."""
    print("\n" + "="*60)
    print("RAG is ready! Ask questions about the scraped website.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\nSearching...")
        result   = qa_chain.invoke({"query": question})
        answer   = result["result"]
        sources  = result.get("source_documents", [])

        print(f"\nAnswer:\n{answer}")

        if sources:
            seen = set()
            print("\nSources:")
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                if src not in seen:
                    print(f"  • {src}")
                    seen.add(src)
        print()


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Website RAG Pipeline"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Root URL to crawl (e.g. https://example.com)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum pages to crawl (default: 30)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Token chunk size (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap (default: 50)"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Skip scraping; load existing ChromaDB and go straight to QA"
    )
    args = parser.parse_args()

    embeddings = get_embedding_model()

    # ── If loading existing DB, skip scraping ──
    if args.load_existing:
        if not os.path.exists(CHROMA_DIR):
            print(f"[ERROR] No existing ChromaDB found at '{CHROMA_DIR}'.")
            print("Run without --load-existing first to build the index.")
            return
        vectordb = load_vector_store(embeddings)

    else:
        # Require URL when scraping
        if not args.url:
            parser.error("--url is required unless --load-existing is set.")

        # Step 1: Crawl
        docs = crawl_website(args.url, max_pages=args.max_pages)
        if not docs:
            print("[ERROR] No pages were scraped. Check the URL and try again.")
            return

        # Step 2: Chunk
        chunks = chunk_documents(
            docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        # Step 3: Embed + Store
        vectordb = build_vector_store(chunks, embeddings)

    # Step 4: QA loop
    qa_chain = build_qa_chain(vectordb)
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()

# https://quotes.toscrape.com/