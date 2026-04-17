import os
import re
import time
import argparse
from urllib.parse import urljoin, urlparse
from collections import deque

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from pinecone import Pinecone

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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in .env file.")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "web-voice")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

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
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

    except Exception as e:
        print(f"  [WARN] Failed {url}: {e}")
        return "", [], []

    for tag in soup(["script", "style", "noscript", "head",
                     "nav", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    text = clean_text(soup.get_text(separator=" "))

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_href = urljoin(url, href).split("#")[0]
        if is_same_domain(url, abs_href):
            links.append(abs_href)

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
# 3. CHUNKING  ✅ Change 2: chunk_size=1000, chunk_overlap=200
# ──────────────────────────────────────────────

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")
    return chunks

# ──────────────────────────────────────────────
# 4. EMBEDDINGS + PINECONE
# ──────────────────────────────────────────────

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def build_vector_store(chunks, embeddings):
    print("Uploading embeddings to Pinecone...")
    vectordb = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    print("Embeddings stored in Pinecone successfully.")
    return vectordb


def load_vector_store(embeddings):
    print("Loading existing Pinecone index...")
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

# ──────────────────────────────────────────────
# 5. QA  ✅ Change 1: Improved anti-hallucination RAG prompt
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise question-answering assistant. Your job is to answer the user's question using ONLY the information provided in the context below.

STRICT RULES you must follow:
1. Answer ONLY using facts explicitly stated in the context. Do not add any information from your own training or general knowledge.
2. If the context does not contain enough information to answer the question fully, say: "I don't have enough information in the provided content to answer this fully." Then share only what the context does support.
3. If the context contains NO relevant information at all, say: "The provided content does not contain information about this topic." Do not attempt to answer.
4. Never guess, assume, or infer details that are not clearly written in the context.
5. Do not say things like "based on general knowledge" or "typically" or "usually" — every sentence in your answer must be traceable to the context.
6. Keep your answer clear, concise, and directly relevant to the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (based strictly on the context above):
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
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--load-existing", action="store_true")
    args = parser.parse_args()

    embeddings = get_embedding_model()

    if args.load_existing:
        vectordb = load_vector_store(embeddings)
    else:
        url = input("Enter the website URL to scrape: ").strip()
        if not url:
            print("No URL provided. Exiting.")
            return

        docs = crawl_website(url, args.max_pages)
        if not docs:
            print("No data scraped.")
            return

        chunks = chunk_documents(docs)
        vectordb = build_vector_store(chunks, embeddings)

    chain = build_qa_chain(vectordb)
    interactive_qa(chain)


if __name__ == "__main__":
    main()