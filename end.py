import os
import re
import time
import asyncio
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

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

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Web RAG Voice API",
    description="Scrapes a website, builds a vector store, and answers queries via RAG.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class PasteURLRequest(BaseModel):
    url: str
    max_pages: Optional[int] = 5

class PasteURLResponse(BaseModel):
    status: str
    url: str
    pages_scraped: int
    chunks_stored: int
    message: str

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

# ──────────────────────────────────────────────
# 1. Selenium Driver
# ──────────────────────────────────────────────
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# ──────────────────────────────────────────────
# 2. Scraper
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


def crawl_website(start_url: str, max_pages: int = 5):
    driver = get_driver()
    visited = set()
    queue = deque([start_url])
    docs = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue

        visited.add(url)
        text, links, images = scrape_page(driver, url)
        if not text:
            continue

        if images:
            text += "\n\n" + "\n".join(images)

        docs.append(Document(page_content=text, metadata={"source": url}))

        for link in links:
            if link not in visited:
                queue.append(link)

    driver.quit()
    return docs

# ──────────────────────────────────────────────
# 3. Chunking
# ──────────────────────────────────────────────
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# ──────────────────────────────────────────────
# 4. Embeddings + Pinecone
# ──────────────────────────────────────────────
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(chunks, embeddings):
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )


def load_vector_store(embeddings):
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

# ──────────────────────────────────────────────
# 5. RAG Prompt + QA Chain
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


def build_qa_chain(vectordb, k: int = 5):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )

# ──────────────────────────────────────────────
# 6. API Endpoints
# ──────────────────────────────────────────────

@app.get("/health", summary="Health Check")
async def health():
    """Returns API health status and confirms environment variables are loaded."""
    return {
        "status": "healthy",
        "pinecone_index": PINECONE_INDEX_NAME,
        "google_api_key_set": bool(GOOGLE_API_KEY),
        "pinecone_api_key_set": bool(PINECONE_API_KEY),
    }


@app.post("/paste_url", response_model=PasteURLResponse, summary="Scrape & Ingest Website")
async def paste_url(request: PasteURLRequest):
    """
    Accepts a website URL, crawls it (up to max_pages), cleans the content,
    chunks it, and stores embeddings in Pinecone.
    """
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        # Run blocking crawl in a thread pool to avoid blocking the event loop
        docs = await asyncio.get_event_loop().run_in_executor(
            None, crawl_website, url, request.max_pages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

    if not docs:
        raise HTTPException(status_code=422, detail="No content could be scraped from the provided URL.")

    try:
        chunks = chunk_documents(docs)
        embeddings = get_embedding_model()
        await asyncio.get_event_loop().run_in_executor(
            None, build_vector_store, chunks, embeddings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store build failed: {str(e)}")

    return PasteURLResponse(
        status="success",
        url=url,
        pages_scraped=len(docs),
        chunks_stored=len(chunks),
        message=f"Successfully scraped {len(docs)} page(s), created {len(chunks)} chunks, and stored embeddings in Pinecone index '{PINECONE_INDEX_NAME}'."
    )


@app.post("/query", response_model=QueryResponse, summary="Query the Vector Store")
async def query(request: QueryRequest):
    """
    Accepts a natural language query, retrieves relevant chunks from Pinecone,
    and returns an answer generated by Gemini using the RAG pipeline.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        embeddings = get_embedding_model()
        vectordb = load_vector_store(embeddings)
        chain = build_qa_chain(vectordb, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store or build QA chain: {str(e)}")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: chain.invoke({"query": request.query})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

    sources = list({doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])})

    return QueryResponse(
        query=request.query,
        answer=result["result"],
        sources=sources
    )