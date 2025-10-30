import os
import re
import hashlib
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pypdf import PdfReader
from pypdf.errors import PdfReadError
import docx
from docx.opc.exceptions import PackageNotFoundError
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding function once at module level
EMB = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _file_to_text(path: str) -> str:
    """
    Extract text content from various file formats.
    
    Args:
        path: Path to the file to process
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
        IOError: If file cannot be read
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".pdf":
            return _extract_pdf(path)
        elif ext in [".txt", ".md"]:
            return _extract_text(path)
        elif ext == ".docx":
            return _extract_docx(path)
        elif ext in [".html", ".htm"]:
            return _extract_html(path)
        else:
            # Attempt to read as plain text for unknown extensions
            logger.warning(f"Unknown extension {ext}, attempting to read as text")
            return _extract_text(path)
    except Exception as e:
        logger.error(f"Error extracting text from {path}: {e}")
        raise IOError(f"Failed to extract text from {path}: {e}") from e


def _extract_pdf(path: str) -> str:
    """Extract text from PDF file."""
    try:
        reader = PdfReader(path)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {path}: {e}")
        return "\n".join(text_parts)
    except PdfReadError as e:
        raise IOError(f"Invalid or corrupted PDF file: {path}") from e


def _extract_text(path: str) -> str:
    """Extract text from plain text or markdown files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_docx(path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    except PackageNotFoundError as e:
        raise IOError(f"Invalid or corrupted DOCX file: {path}") from e


def _extract_html(path: str) -> str:
    """Extract text from HTML file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    return soup.get_text(separator=" ", strip=True)


def _chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If size or overlap parameters are invalid
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if overlap >= size:
        raise ValueError("Overlap must be less than chunk size")
    
    if not text or not text.strip():
        return []
    
    words = re.split(r"\s+", text.strip())
    chunks = []
    i = 0
    
    while i < len(words):
        end_idx = min(i + size, len(words))
        chunk = " ".join(words[i:end_idx])
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        # Move forward by (size - overlap) words
        i += size - overlap
        
        # Prevent infinite loop if we're at the end
        if end_idx == len(words):
            break
    
    return chunks


class RAG:
    """
    Retrieval-Augmented Generation system using ChromaDB for vector storage.
    """
    
    def __init__(self, persist_dir: str = "./chroma"):
        """
        Initialize RAG system.
        
        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            logger.info(f"Initialized ChromaDB at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _collection(self, namespace: Optional[str] = None):
        """
        Get or create a collection for the given namespace.
        
        Args:
            namespace: Optional namespace for organizing collections
            
        Returns:
            ChromaDB collection
        """
        # Sanitize namespace to ensure valid collection name
        ns = (namespace or "default").lower()
        ns = re.sub(r'[^a-z0-9_-]', '_', ns)
        name = f"kb_{ns}"
        
        try:
            return self.client.get_or_create_collection(
                name=name, 
                embedding_function=EMB
            )
        except Exception as e:
            logger.error(f"Failed to get/create collection {name}: {e}")
            raise

    def ingest(
        self, 
        paths: List[str], 
        namespace: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120
    ) -> Dict[str, Any]:
        """
        Ingest documents into the knowledge base.
        
        Args:
            paths: List of file paths to ingest
            namespace: Optional namespace for organizing documents
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not paths:
            logger.warning("No paths provided for ingestion")
            return {"total_chunks": 0, "successful_files": 0, "failed_files": 0, "errors": []}
        
        col = self._collection(namespace)
        total_chunks = 0
        successful_files = 0
        failed_files = 0
        errors = []
        
        for path in paths:
            try:
                text = _file_to_text(path)
                
                if not text or not text.strip():
                    logger.warning(f"No text extracted from {path}")
                    continue
                
                chunks = _chunk(text, size=chunk_size, overlap=chunk_overlap)
                
                if not chunks:
                    logger.warning(f"No chunks created from {path}")
                    continue
                
                # Batch add chunks for better performance
                ids = []
                documents = []
                metadatas = []
                
                for idx, chunk in enumerate(chunks):
                    uid = hashlib.sha256(f"{path}::{idx}".encode()).hexdigest()
                    ids.append(uid)
                    documents.append(chunk)
                    metadatas.append({
                        "source": os.path.basename(path),
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    })
                
                col.add(ids=ids, documents=documents, metadatas=metadatas)
                total_chunks += len(chunks)
                successful_files += 1
                logger.info(f"Ingested {len(chunks)} chunks from {path}")
                
            except Exception as e:
                failed_files += 1
                error_msg = f"Failed to ingest {path}: {str(e)}"
                logger.error(error_msg)
                errors.append({"file": path, "error": str(e)})
        
        result = {
            "total_chunks": total_chunks,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "errors": errors
        }
        
        logger.info(f"Ingestion complete: {result}")
        return result

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        namespace: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Optional namespace to search in
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            
        Returns:
            List of retrieved documents with metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        
        try:
            col = self._collection(namespace)
            
            # Check if collection has any documents
            count = col.count()
            if count == 0:
                logger.warning(f"Collection is empty for namespace: {namespace}")
                return []
            
            # Adjust top_k if it exceeds collection size
            actual_k = min(top_k, count)
            
            res = col.query(
                query_texts=[query], 
                n_results=actual_k, 
                include=["documents", "metadatas", "distances"]
            )
            
            docs = []
            if res and res.get("documents") and res["documents"][0]:
                docs_raw = res["documents"][0]
                metas = res["metadatas"][0]
                dists = res.get("distances", [[0] * len(docs_raw)])[0]
                
                # Convert distances to similarity scores
                sims = [1 / (1 + d) for d in dists]
                
                for chunk, meta, sim in zip(docs_raw, metas, sims):
                    if sim >= min_score:
                        docs.append({
                            "text": chunk,
                            "source": meta.get("source", "unknown"),
                            "chunk_index": meta.get("chunk_index", 0),
                            "score": round(sim, 4)
                        })
            
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    def delete_namespace(self, namespace: Optional[str] = None) -> bool:
        """
        Delete an entire namespace (collection).
        
        Args:
            namespace: Namespace to delete
            
        Returns:
            True if successful
        """
        ns = (namespace or "default").lower()
        ns = re.sub(r'[^a-z0-9_-]', '_', ns)
        name = f"kb_{ns}"
        
        try:
            self.client.delete_collection(name=name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False

    def list_namespaces(self) -> List[str]:
        """
        List all available namespaces.
        
        Returns:
            List of namespace names
        """
        try:
            collections = self.client.list_collections()
            namespaces = []
            for col in collections:
                if col.name.startswith("kb_"):
                    namespace = col.name[3:]  # Remove 'kb_' prefix
                    namespaces.append(namespace)
            return namespaces
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAG(persist_dir="./my_knowledge_base")
    
    # Ingest documents
    result = rag.ingest(["document.pdf", "notes.txt"], namespace="research")
    print(f"Ingested {result['total_chunks']} chunks from {result['successful_files']} files")
    
    # Retrieve relevant documents
    docs = rag.retrieve("What is machine learning?", top_k=3, namespace="research")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Source: {doc['source']} (Score: {doc['score']})")
        print(f"   {doc['text'][:200]}...")