# app/web_tools.py
import asyncio
import logging
from typing import Tuple, List, Optional, Dict, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RESULTS = 3
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_CONTENT_LENGTH = 2000
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_RETRIES = 2


def _is_valid_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


def _clean_text(soup: BeautifulSoup) -> str:
    """
    Extract and clean text from BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Cleaned text content
    """
    # Remove script, style, and other non-content elements
    for element in soup(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav']):
        element.decompose()
    
    # Get text and clean it
    text = soup.get_text(separator=" ", strip=True)
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    return text


async def _fetch_url_content(
    client: httpx.AsyncClient,
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_length: int = DEFAULT_MAX_CONTENT_LENGTH
) -> Optional[str]:
    """
    Fetch and extract text content from a URL.
    
    Args:
        client: HTTP client instance
        url: URL to fetch
        timeout: Request timeout in seconds
        max_length: Maximum content length to return
        
    Returns:
        Extracted text content or None if failed
    """
    try:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        response = await client.get(
            url,
            follow_redirects=True,
            headers=headers,
            timeout=timeout
        )
        
        # Check response status
        if response.status_code != 200:
            logger.warning(f"Non-200 status code {response.status_code} for {url}")
            return None
        
        # Check content type
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            logger.warning(f"Unsupported content type {content_type} for {url}")
            return None
        
        # Parse and extract text
        soup = BeautifulSoup(response.text, "html.parser")
        text = _clean_text(soup)
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient content extracted from {url}")
            return None
        
        # Truncate to max length
        return text[:max_length] if len(text) > max_length else text
        
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error for {url}: {e}")
    except httpx.RequestError as e:
        logger.warning(f"Request error for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
    
    return None


async def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    backend: str = "api"
) -> List[Dict[str, str]]:
    """
    Search using DuckDuckGo and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        backend: DuckDuckGo backend to use ('api', 'html', 'lite')
        
    Returns:
        List of search results with 'title', 'href', and 'body' keys
    """
    if not query or not query.strip():
        logger.warning("Empty search query provided")
        return []
    
    if max_results <= 0:
        logger.warning(f"Invalid max_results: {max_results}, using default")
        max_results = DEFAULT_MAX_RESULTS
    
    results = []
    
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query.strip(),
                max_results=max_results,
                backend=backend
            )
            
            for result in search_results:
                if not isinstance(result, dict):
                    continue
                
                href = result.get("href", "")
                if not href or not _is_valid_url(href):
                    continue
                
                results.append({
                    "title": result.get("title", "No Title"),
                    "href": href,
                    "body": result.get("body", "")
                })
                
                if len(results) >= max_results:
                    break
        
        logger.info(f"Found {len(results)} search results for query: {query[:50]}...")
        return results
        
    except DuckDuckGoSearchException as e:
        logger.error(f"DuckDuckGo search error for query '{query}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error during search for query '{query}': {e}")
    
    return []


async def web_answer(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    backend: str = "api"
) -> Tuple[str, List[str]]:
    """
    Search for a query and fetch content from top results.
    Returns concatenated context and list of citation URLs.
    On ANY error, returns empty context & no citations for graceful degradation.
    
    Args:
        query: Search query string
        max_results: Maximum number of search results to process
        timeout: Request timeout in seconds
        max_content_length: Maximum content length per URL
        backend: DuckDuckGo backend to use
        
    Returns:
        Tuple of (concatenated_context, list_of_citation_urls)
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to web_answer")
        return "", []
    
    try:
        # Perform search
        search_results = await web_search(query, max_results=max_results, backend=backend)
        
        if not search_results:
            logger.info(f"No search results found for query: {query[:50]}...")
            return "", []
        
        # Extract URLs
        urls = [result["href"] for result in search_results if result.get("href")]
        
        if not urls:
            logger.warning("No valid URLs found in search results")
            return "", []
        
        # Fetch content from URLs
        contexts = []
        citations = []
        
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        ) as client:
            # Create tasks for concurrent fetching
            tasks = [
                _fetch_url_content(client, url, timeout, max_content_length)
                for url in urls
            ]
            
            # Fetch all URLs concurrently with timeout
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during concurrent fetch: {e}")
                return "", []
            
            # Process results
            for url, content in zip(urls, results):
                # Skip exceptions or None results
                if isinstance(content, Exception):
                    logger.warning(f"Exception fetching {url}: {content}")
                    continue
                
                if content and isinstance(content, str):
                    contexts.append(content)
                    citations.append(url)
        
        # Return results
        if contexts:
            combined_context = "\n\n".join(contexts)
            logger.info(f"Successfully fetched {len(contexts)} contexts for query: {query[:50]}...")
            return combined_context, citations
        else:
            logger.warning(f"No content extracted for query: {query[:50]}...")
            return "", []
            
    except Exception as e:
        # Catch-all for any unexpected errors - graceful degradation
        logger.error(f"Unexpected error in web_answer for query '{query}': {e}")
        return "", []


async def web_answer_with_snippets(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: int = DEFAULT_TIMEOUT,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    backend: str = "api"
) -> Dict[str, Any]:
    """
    Enhanced version that returns both search snippets and fetched content.
    Useful when you want both the search result summaries and full page content.
    
    Args:
        query: Search query string
        max_results: Maximum number of search results to process
        timeout: Request timeout in seconds
        max_content_length: Maximum content length per URL
        backend: DuckDuckGo backend to use
        
    Returns:
        Dictionary with 'context', 'citations', 'snippets', and 'metadata'
    """
    try:
        # Perform search
        search_results = await web_search(query, max_results=max_results, backend=backend)
        
        if not search_results:
            return {
                "context": "",
                "citations": [],
                "snippets": [],
                "metadata": {"query": query, "results_found": 0}
            }
        
        # Extract URLs and snippets
        urls = [result["href"] for result in search_results]
        snippets = [
            {
                "title": result.get("title", "No Title"),
                "body": result.get("body", ""),
                "url": result.get("href", "")
            }
            for result in search_results
        ]
        
        # Fetch full content
        contexts = []
        citations = []
        
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        ) as client:
            tasks = [
                _fetch_url_content(client, url, timeout, max_content_length)
                for url in urls
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, content in zip(urls, results):
                if content and isinstance(content, str):
                    contexts.append(content)
                    citations.append(url)
        
        combined_context = "\n\n".join(contexts) if contexts else ""
        
        return {
            "context": combined_context,
            "citations": citations,
            "snippets": snippets,
            "metadata": {
                "query": query,
                "results_found": len(search_results),
                "content_fetched": len(contexts)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in web_answer_with_snippets: {e}")
        return {
            "context": "",
            "citations": [],
            "snippets": [],
            "metadata": {"query": query, "error": str(e)}
        }


# Synchronous wrapper for backwards compatibility
def web_answer_sync(query: str, **kwargs) -> Tuple[str, List[str]]:
    """
    Synchronous wrapper for web_answer.
    Useful when you can't use async/await in your code.
    
    Args:
        query: Search query string
        **kwargs: Additional arguments to pass to web_answer
        
    Returns:
        Tuple of (context, citations)
    """
    try:
        return asyncio.run(web_answer(query, **kwargs))
    except Exception as e:
        logger.error(f"Error in synchronous web_answer: {e}")
        return "", []


# Example usage
if __name__ == "__main__":
    async def main():
        # Basic usage
        context, citations = await web_answer("Python web scraping best practices")
        print(f"Context length: {len(context)}")
        print(f"Citations: {citations}")
        
        # Enhanced usage with snippets
        result = await web_answer_with_snippets(
            "Machine learning tutorials",
            max_results=3,
            max_content_length=1500
        )
        print(f"\nMetadata: {result['metadata']}")
        print(f"Snippets: {len(result['snippets'])}")
        
        # Synchronous usage
        sync_context, sync_citations = web_answer_sync("Async Python tutorial")
        print(f"\nSync result - Context length: {len(sync_context)}")
    
    # Run the example
    asyncio.run(main())