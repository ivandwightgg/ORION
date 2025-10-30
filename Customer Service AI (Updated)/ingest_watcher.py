import os
import json
import time
import hashlib
import asyncio
import logging
import re
from typing import Any
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import yaml
import aiofiles
import aiofiles.os
from filelock import FileLock

from .rag import RAG
from .web_utils import fetch_page_text

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration safely
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    ING = CONFIG["ingest"]
except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

STATE_FILE = Path(ING["root"]) / ".ingested_state.json"
LOCK_FILE = Path(ING["root"]) / ".ingested_state.lock"
STATE_LOCK = asyncio.Lock()

# Constants
DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
URL_PATTERN = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
HASH_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for faster hashing
MAX_URL_LENGTH = 2048
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class IngestionError(Exception):
    """Custom exception for ingestion errors."""
    pass


async def _load_state() -> dict[str, dict[str, Any]]:
    """
    Load the ingestion state from disk with file locking.
    
    Returns:
        Dictionary containing the state, or empty dict if file doesn't exist.
    """
    if not STATE_FILE.exists():
        logger.info("State file does not exist, starting with empty state")
        return {}
    
    lock = FileLock(str(LOCK_FILE), timeout=10)
    
    try:
        with lock:
            async with aiofiles.open(STATE_FILE, "r", encoding="utf-8") as f:
                content = await f.read()
                if not content.strip():
                    logger.warning("State file is empty")
                    return {}
                state = json.loads(content)
                logger.info(f"Loaded state with {len(state)} entries")
                return state
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in state file: {e}")
        # Backup corrupted file
        backup_path = STATE_FILE.with_suffix(f".backup.{int(time.time())}")
        try:
            await aiofiles.os.rename(STATE_FILE, backup_path)
            logger.info(f"Backed up corrupted state file to {backup_path}")
        except OSError:
            pass
        return {}
    except OSError as e:
        logger.error(f"Error loading state file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading state: {e}", exc_info=True)
        return {}


async def _save_state(state: dict[str, dict[str, Any]]) -> None:
    """
    Save the ingestion state to disk atomically with file locking.
    
    Args:
        state: The state dictionary to save.
        
    Raises:
        IngestionError: If saving fails after retries.
    """
    lock = FileLock(str(LOCK_FILE), timeout=10)
    
    for attempt in range(MAX_RETRIES):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = STATE_FILE.with_suffix(f".tmp.{os.getpid()}")
            
            with lock:
                async with aiofiles.open(tmp_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(state, indent=2, sort_keys=True))
                    await f.flush()
                
                # Atomic replace
                await aiofiles.os.replace(tmp_file, STATE_FILE)
                logger.info(f"Saved state with {len(state)} entries")
                return
                
        except OSError as e:
            logger.error(f"Error saving state file (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise IngestionError(f"Failed to save state after {MAX_RETRIES} attempts") from e
        except Exception as e:
            logger.error(f"Unexpected error saving state: {e}", exc_info=True)
            raise IngestionError("Failed to save state") from e
        finally:
            # Clean up temp file if it exists
            try:
                if tmp_file.exists():
                    await aiofiles.os.remove(tmp_file)
            except OSError:
                pass


async def _hash_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file asynchronously.
    
    Args:
        path: Path to the file to hash.
        
    Returns:
        Hexadecimal hash string.
        
    Raises:
        IngestionError: If file cannot be read.
    """
    h = hashlib.sha256()  # SHA-256 is more secure than SHA-1
    
    try:
        file_size = path.stat().st_size
        logger.debug(f"Hashing file {path} ({file_size} bytes)")
        
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(HASH_CHUNK_SIZE):
                h.update(chunk)
                
        hash_value = h.hexdigest()
        logger.debug(f"Hash for {path}: {hash_value}")
        return hash_value
        
    except OSError as e:
        logger.error(f"Error hashing file {path}: {e}")
        raise IngestionError(f"Cannot hash file {path}") from e


async def _ingest_files_in_dir(
    rag: RAG,
    dir_path: Path,
    namespace: str,
    state: dict[str, dict[str, Any]]
) -> int:
    """
    Ingest document files from a directory.
    
    Args:
        rag: RAG instance for ingestion.
        dir_path: Directory to scan for files.
        namespace: Namespace for the documents.
        state: Current state dictionary.
        
    Returns:
        Number of files ingested.
    """
    count = 0
    errors = 0
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return 0
    
    if not dir_path.is_dir():
        logger.warning(f"Path is not a directory: {dir_path}")
        return 0
    
    logger.info(f"Scanning directory for files: {dir_path} (namespace: {namespace})")
    
    try:
        for root, _, files in os.walk(dir_path):
            root_path = Path(root)
            
            for name in files:
                file_path = root_path / name
                ext = file_path.suffix.lower()
                
                if ext not in DOC_EXTS:
                    continue
                
                # Validate file is readable
                if not file_path.is_file():
                    logger.warning(f"Skipping non-file: {file_path}")
                    continue
                
                if not os.access(file_path, os.R_OK):
                    logger.warning(f"Skipping unreadable file: {file_path}")
                    continue
                
                key = f"file::{namespace}::{file_path.absolute()}"
                
                try:
                    sha = await _hash_file(file_path)
                    entry = state.get(key)
                    
                    # Skip if already ingested with same hash
                    if entry and entry.get("sha") == sha:
                        logger.debug(f"File unchanged, skipping: {file_path}")
                        continue
                    
                    # Ingest file (handle both sync and async)
                    logger.info(f"Ingesting file: {file_path}")
                    if asyncio.iscoroutinefunction(rag.ingest):
                        await rag.ingest([str(file_path)], namespace=namespace)
                    else:
                        await asyncio.to_thread(rag.ingest, [str(file_path)], namespace=namespace)
                    
                    state[key] = {
                        "sha": sha,
                        "ts": time.time(),
                        "date": datetime.now().isoformat(),
                        "size": file_path.stat().st_size
                    }
                    count += 1
                    logger.info(f"Successfully ingested file: {file_path}")
                    
                except IngestionError as e:
                    logger.error(f"Ingestion error for {file_path}: {e}")
                    errors += 1
                    state[key] = {
                        "error": str(e),
                        "ts": time.time(),
                        "date": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Unexpected error ingesting file {file_path}: {e}", exc_info=True)
                    errors += 1
                    continue
    
    except OSError as e:
        logger.error(f"Error walking directory {dir_path}: {e}")
    
    logger.info(f"Completed file ingestion: {count} ingested, {errors} errors")
    return count


async def _ingest_links_in_dir(
    rag: RAG,
    dir_path: Path,
    namespace: str,
    state: dict[str, dict[str, Any]]
) -> int:
    """
    Ingest web links from text files in a directory.
    
    Args:
        rag: RAG instance for ingestion.
        dir_path: Directory to scan for link files.
        namespace: Namespace for the documents.
        state: Current state dictionary.
        
    Returns:
        Number of URLs ingested.
    """
    count = 0
    errors = 0
    cache_dir = Path(ING["links_cache_dir"])
    
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create cache directory {cache_dir}: {e}")
        return 0
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return 0
    
    if not dir_path.is_dir():
        logger.warning(f"Path is not a directory: {dir_path}")
        return 0
    
    logger.info(f"Scanning directory for links: {dir_path} (namespace: {namespace})")
    
    try:
        for root, _, files in os.walk(dir_path):
            root_path = Path(root)
            
            for name in files:
                if not name.lower().endswith(".txt"):
                    continue
                
                file_path = root_path / name
                
                if not file_path.is_file() or not os.access(file_path, os.R_OK):
                    logger.warning(f"Skipping unreadable file: {file_path}")
                    continue
                
                key_file = f"linksfile::{namespace}::{file_path.absolute()}"
                
                try:
                    mtime = file_path.stat().st_mtime
                    entry = state.get(key_file)
                    
                    # Skip if file hasn't been modified
                    if entry and entry.get("mtime") == mtime:
                        logger.debug(f"Links file unchanged, skipping: {file_path}")
                        continue
                    
                    logger.info(f"Processing links file: {file_path}")
                    
                    # Read URLs from file
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        urls = [
                            ln.strip() 
                            for ln in content.splitlines() 
                            if ln.strip() and not ln.strip().startswith("#")
                        ]
                    
                    logger.info(f"Found {len(urls)} URLs in {file_path}")
                    
                    for url in urls:
                        # Validate URL
                        if not URL_PATTERN.match(url):
                            logger.warning(f"Invalid URL format: {url}")
                            continue
                        
                        if len(url) > MAX_URL_LENGTH:
                            logger.warning(f"URL too long (>{MAX_URL_LENGTH} chars): {url[:100]}...")
                            continue
                        
                        key = f"url::{namespace}::{url}"
                        
                        # Skip if already successfully ingested
                        if key in state and state[key].get("ok"):
                            logger.debug(f"URL already ingested, skipping: {url}")
                            continue
                        
                        try:
                            logger.info(f"Fetching URL: {url}")
                            text = await fetch_page_text(url)
                            
                            if not text or not text.strip():
                                logger.warning(f"Empty content from URL: {url}")
                                state[key] = {
                                    "ok": False,
                                    "error": "Empty content",
                                    "ts": time.time(),
                                    "date": datetime.now().isoformat()
                                }
                                continue
                            
                            # Cache the content
                            sha = hashlib.sha256(url.encode()).hexdigest()
                            cache_path = cache_dir / f"{sha}.txt"
                            
                            async with aiofiles.open(cache_path, "w", encoding="utf-8") as cf:
                                await cf.write(text)
                            
                            # Ingest cached content
                            logger.info(f"Ingesting content from URL: {url}")
                            if asyncio.iscoroutinefunction(rag.ingest):
                                await rag.ingest([str(cache_path)], namespace=namespace)
                            else:
                                await asyncio.to_thread(rag.ingest, [str(cache_path)], namespace=namespace)
                            
                            state[key] = {
                                "ok": True,
                                "ts": time.time(),
                                "date": datetime.now().isoformat(),
                                "cache": str(cache_path),
                                "size": len(text)
                            }
                            count += 1
                            logger.info(f"Successfully ingested URL: {url}")
                            
                        except Exception as e:
                            logger.error(f"Error fetching/ingesting URL {url}: {e}")
                            errors += 1
                            state[key] = {
                                "ok": False,
                                "error": str(e)[:500],  # Limit error message length
                                "ts": time.time(),
                                "date": datetime.now().isoformat()
                            }
                    
                    # Update file state
                    state[key_file] = {
                        "mtime": mtime,
                        "ts": time.time(),
                        "date": datetime.now().isoformat(),
                        "url_count": len(urls)
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing links file {file_path}: {e}", exc_info=True)
                    errors += 1
                    continue
    
    except OSError as e:
        logger.error(f"Error walking directory {dir_path}: {e}")
    
    logger.info(f"Completed link ingestion: {count} ingested, {errors} errors")
    return count


async def scan_all(rag: RAG) -> dict[str, int]:
    """
    Scan all configured directories and ingest new content.
    
    Args:
        rag: RAG instance for ingestion.
        
    Returns:
        Dictionary with counts of ingested documents and links.
    """
    async with STATE_LOCK:
        logger.info("Starting scan of all directories")
        start_time = time.time()
        
        state = await _load_state()
        totals = {"docs": 0, "links": 0}
        
        uploads_dir = Path(ING["uploads_dir"])
        links_dir = Path(ING["links_dir"])
        messages_dir = Path(ING["messages_dir"])
        
        def ns_dirs(base_dir: Path) -> list[tuple[Path, str]]:
            """
            Get namespace directories from a base directory.
            
            Args:
                base_dir: Base directory to scan.
                
            Returns:
                List of (path, namespace) tuples.
            """
            if not base_dir.exists():
                logger.warning(f"Base directory does not exist: {base_dir}")
                return []
            
            items = []
            try:
                for entry in base_dir.iterdir():
                    if entry.is_dir():
                        items.append((entry, entry.name))
            except OSError as e:
                logger.error(f"Error scanning directory {base_dir}: {e}")
            
            # Include base dir as default namespace bucket
            default_ns = ING.get("namespace_default", "default")
            items.append((base_dir, default_ns))
            
            logger.info(f"Found {len(items)} namespace directories in {base_dir}")
            return items
        
        # Process uploads directory
        logger.info("Processing uploads directory")
        for dir_path, ns in ns_dirs(uploads_dir):
            totals["docs"] += await _ingest_files_in_dir(rag, dir_path, ns, state)
        
        # Process messages directory
        logger.info("Processing messages directory")
        for dir_path, ns in ns_dirs(messages_dir):
            totals["docs"] += await _ingest_files_in_dir(rag, dir_path, ns, state)
        
        # Process links directory
        logger.info("Processing links directory")
        for dir_path, ns in ns_dirs(links_dir):
            totals["links"] += await _ingest_links_in_dir(rag, dir_path, ns, state)
        
        # Save state
        await _save_state(state)
        
        elapsed = time.time() - start_time
        logger.info(f"Scan complete in {elapsed:.2f}s: {totals}")
        return totals


async def run_watcher(rag: RAG, interval: int = 60) -> None:
    """
    Run the ingestion watcher loop.
    
    Args:
        rag: RAG instance for ingestion.
        interval: Seconds between scans (minimum 3).
    """
    interval = max(3, interval)
    logger.info(f"Starting ingestion watcher (interval: {interval}s)")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            await scan_all(rag)
            consecutive_errors = 0  # Reset on success
            
        except IngestionError as e:
            consecutive_errors += 1
            logger.error(f"Ingestion error in watcher loop: {e}")
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Unexpected error in watcher loop: {e}", exc_info=True)
        
        # Backoff if too many consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            logger.critical(
                f"Too many consecutive errors ({consecutive_errors}), "
                f"backing off for {interval * 5}s"
            )
            await asyncio.sleep(interval * 5)
            consecutive_errors = 0
        else:
            await asyncio.sleep(interval)


async def cleanup_old_cache(max_age_days: int = 30) -> int:
    """
    Clean up old cached files.
    
    Args:
        max_age_days: Maximum age in days for cached files.
        
    Returns:
        Number of files deleted.
    """
    cache_dir = Path(ING["links_cache_dir"])
    
    if not cache_dir.exists():
        return 0
    
    logger.info(f"Cleaning up cache files older than {max_age_days} days")
    count = 0
    max_age_seconds = max_age_days * 24 * 3600
    cutoff_time = time.time() - max_age_seconds
    
    try:
        for file_path in cache_dir.glob("*.txt"):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    await aiofiles.os.remove(file_path)
                    count += 1
                    logger.debug(f"Deleted old cache file: {file_path}")
            except OSError as e:
                logger.error(f"Error deleting cache file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
    
    logger.info(f"Cleaned up {count} old cache files")
    return count