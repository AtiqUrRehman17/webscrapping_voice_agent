import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ✅ Updated import (no deprecation)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ✅ Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ──────────────────────────────────────────────
# 0. Load environment
# ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file.")

CHROMA_DIR = "./chroma_db"
COLLECTION = "website_rag"

# ──────────────────────────────────────────────
# 1. SELENIUM DRIVER
# ──────────────────────────────────────────────

def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# ──────────────────────────────────────────────
# 2. SCRAPER
# ──────────────────────────────────────────────

def is_same_domain(base_url: str, link: str) -> bool:
    base_netloc = urlparse(base_url).netloc
    link_netloc = urlparse(link).netloc
    return link_netloc == "" or link_netloc == base_netloc


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_page(driver, url: str):
    try:
        driver.get(url)

        # wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # scroll to load lazy content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")

    except Exception as e:
        print(f"  [WARN] Failed {url}: {e}")
        return "", [], []

    # remove noise
    for tag in soup(["script", "style", "noscript", "head",
                     "nav", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    text = clean_text(soup.get_text(separator=" "))

    # links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        abs_href = urljoin(url, href).split("#")[0]

        if is_same_domain(url, abs_href):
            links.append(abs_href)

    # images
    images = []
    for img in soup.find_all("img"):
        alt = img.get("alt", "").strip()
        src = img.get("src", "").strip()

        if alt:
            images.append(f"[IMAGE: {alt}]")
        elif src:
            images.append(f"[IMAGE src: {src}]")

    return text, list(set(links)), images


def crawl_website(start_url: str, max_pages: int = 30):
    print("\nStarting crawl:", start_url)

    driver = get_driver()

    visited = set()
    queue = deque([start_url])
    docs = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue

        print(f"  Scraping [{len(visited)+1}/{max_pages}]: {url}")
        visited.add(url)

        text, links, images = scrape_page(driver, url)

        if not text:
            continue

        if images:
            text += "\n\n" + "\n".join(images)

        docs.append(Document(
            page_content=text,
            metadata={"source": url}
        ))

        for link in links:
            if link not in visited:
                queue.append(link)

    driver.quit()

    print(f"Crawl complete. Pages scraped: {len(docs)}")
    return docs

# ──────────────────────────────────────────────
# 3. CHUNKING
# ──────────────────────────────────────────────

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")
    return chunks

# ──────────────────────────────────────────────
# 4. EMBEDDINGS + DB
# ──────────────────────────────────────────────

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(chunks, embeddings):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION
    )
    vectordb.persist()
    return vectordb


def load_vector_store(embeddings):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )

# ──────────────────────────────────────────────
# 5. QA
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use ONLY the context below.

Context:
{context}

Question: {question}
Answer:
"""
)


def build_qa_chain(vectordb):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )


def interactive_qa(chain):
    while True:
        q = input("\nAsk: ")
        if q.lower() in ["exit", "quit"]:
            break

        result = chain.invoke({"query": q})
        print("\nAnswer:", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata["source"])

# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--load-existing", action="store_true")

    args = parser.parse_args()

    embeddings = get_embedding_model()

    if args.load_existing:
        vectordb = load_vector_store(embeddings)
    else:
        if not args.url:
            raise ValueError("Provide --url")

        docs = crawl_website(args.url, args.max_pages)

        if not docs:
            print("No data scraped.")
            return

        chunks = chunk_documents(docs)
        vectordb = build_vector_store(chunks, embeddings)

    chain = build_qa_chain(vectordb)
    interactive_qa(chain)


if __name__ == "__main__":
    main()