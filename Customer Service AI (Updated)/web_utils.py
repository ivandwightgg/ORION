# web_utils.py
import asyncio
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
import re

import httpx
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 25
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB limit
DEFAULT_MAX_TEXT_LENGTH = 100000  # 100k characters


class FetchError(Exception):
    """Base exception for fetch-related errors."""
    pass


class InvalidURLError(FetchError):
    """Raised when URL is invalid."""
    pass


class ContentTooLargeError(FetchError):
    """Raised when content exceeds size limit."""
    pass


class UnsupportedContentTypeError(FetchError):
    """Raised when content type is not supported."""
    pass


def validate_url(url: str) -> bool:
    """
    Validate URL format and scheme.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid
        
    Raises:
        InvalidURLError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError("URL must be a non-empty string")
    
    url = url.strip()
    
    try:
        result = urlparse(url)
        
        if not result.scheme:
            raise InvalidURLError(f"URL missing scheme: {url}")
        
        if result.scheme not in ('http', 'https'):
            raise InvalidURLError(f"Unsupported URL scheme: {result.scheme}")
        
        if not result.netloc:
            raise InvalidURLError(f"URL missing domain: {url}")
        
        return True
        
    except Exception as e:
        if isinstance(e, InvalidURLError):
            raise
        raise InvalidURLError(f"Invalid URL format: {url}") from e


def clean_html_text(soup: BeautifulSoup, preserve_links: bool = False) -> str:
    """
    Extract and clean text from BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup parsed HTML
        preserve_links: If True, keep link URLs in parentheses
        
    Returns:
        Cleaned text content
    """
    # Remove unwanted elements
    for element in soup(['script', 'style', 'meta', 'link', 'noscript', 
                        'iframe', 'svg', 'path', 'input', 'button']):
        element.decompose()
    
    # Optionally preserve links
    if preserve_links:
        for link in soup.find_all('a', href=True):
            link.string = f"{link.get_text()} ({link['href']})"
    
    # Get text with spacing
    text = soup.get_text(separator=" ", strip=True)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


async def fetch_page_text(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_length: Optional[int] = DEFAULT_MAX_TEXT_LENGTH,
    preserve_links: bool = False,
    raise_on_error: bool = False
) -> str:
    """
    Fetch and extract text content from a web page.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_length: Maximum text length to return (None for unlimited)
        preserve_links: If True, preserve link URLs in the text
        raise_on_error: If True, raise exceptions; if False, return empty string
        
    Returns:
        Extracted text content or empty string on error (if raise_on_error=False)
        
    Raises:
        InvalidURLError: If URL is invalid
        ContentTooLargeError: If content exceeds size limit
        UnsupportedContentTypeError: If content type is not HTML
        httpx.HTTPError: For HTTP-related errors (if raise_on_error=True)
    """
    try:
        # Validate URL
        validate_url(url)
        
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_redirects=5)
        ) as client:
            
            # Fetch the page
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
            except httpx.TimeoutException as e:
                logger.error(f"Timeout fetching {url}: {e}")
                if raise_on_error:
                    raise
                return ""
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code} error for {url}")
                if raise_on_error:
                    raise
                return ""
            except httpx.RequestError as e:
                logger.error(f"Request error for {url}: {e}")
                if raise_on_error:
                    raise
                return ""
            
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if not any(ct in content_type for ct in ["text/html", "application/xhtml", "text/plain"]):
                error_msg = f"Unsupported content type '{content_type}' for {url}"
                logger.warning(error_msg)
                if raise_on_error:
                    raise UnsupportedContentTypeError(error_msg)
                return ""
            
            # Check content size
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_CONTENT_SIZE:
                error_msg = f"Content too large ({content_length} bytes) for {url}"
                logger.warning(error_msg)
                if raise_on_error:
                    raise ContentTooLargeError(error_msg)
                return ""
            
            # Check actual response size
            if len(response.content) > MAX_CONTENT_SIZE:
                error_msg = f"Response too large ({len(response.content)} bytes) for {url}"
                logger.warning(error_msg)
                if raise_on_error:
                    raise ContentTooLargeError(error_msg)
                return ""
            
            # Parse HTML
            try:
                soup = BeautifulSoup(response.text, "html.parser")
            except Exception as e:
                logger.error(f"Error parsing HTML from {url}: {e}")
                if raise_on_error:
                    raise
                return ""
            
            # Extract text
            text = clean_html_text(soup, preserve_links=preserve_links)
            
            # Truncate if needed
            if max_length and len(text) > max_length:
                text = text[:max_length]
                logger.info(f"Truncated text from {url} to {max_length} characters")
            
            logger.info(f"Successfully fetched {len(text)} characters from {url}")
            return text
            
    except (InvalidURLError, ContentTooLargeError, UnsupportedContentTypeError) as e:
        logger.error(f"Fetch error: {e}")
        if raise_on_error:
            raise
        return ""
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        if raise_on_error:
            raise
        return ""


async def fetch_page_metadata(url: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Fetch metadata from a web page (title, description, etc.).
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing page metadata
    """
    metadata = {
        "url": url,
        "title": "",
        "description": "",
        "author": "",
        "keywords": [],
        "canonical_url": "",
        "language": "",
        "success": False
    }
    
    try:
        validate_url(url)
        
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            if soup.title:
                metadata["title"] = soup.title.string.strip() if soup.title.string else ""
            
            # Extract meta tags
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                property_tag = meta.get("property", "").lower()
                content = meta.get("content", "")
                
                if name == "description" or property_tag == "og:description":
                    if not metadata["description"]:
                        metadata["description"] = content
                
                elif name == "author":
                    metadata["author"] = content
                
                elif name == "keywords":
                    metadata["keywords"] = [k.strip() for k in content.split(",")]
                
                elif property_tag == "og:title" and not metadata["title"]:
                    metadata["title"] = content
            
            # Extract canonical URL
            canonical = soup.find("link", rel="canonical")
            if canonical and canonical.get("href"):
                metadata["canonical_url"] = urljoin(url, canonical["href"])
            
            # Extract language
            html_tag = soup.find("html")
            if html_tag:
                metadata["language"] = html_tag.get("lang", "")
            
            metadata["success"] = True
            logger.info(f"Successfully extracted metadata from {url}")
            
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")
    
    return metadata


async def fetch_multiple_pages(
    urls: List[str],
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrent: int = 5,
    **kwargs
) -> Dict[str, str]:
    """
    Fetch text from multiple URLs concurrently.
    
    Args:
        urls: List of URLs to fetch
        timeout: Request timeout in seconds per URL
        max_concurrent: Maximum number of concurrent requests
        **kwargs: Additional arguments to pass to fetch_page_text
        
    Returns:
        Dictionary mapping URLs to their extracted text content
    """
    if not urls:
        logger.warning("No URLs provided to fetch_multiple_pages")
        return {}
    
    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(urls))
    
    results = {}
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(url: str) -> tuple:
        async with semaphore:
            try:
                text = await fetch_page_text(url, timeout=timeout, **kwargs)
                return url, text
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return url, ""
    
    # Fetch all URLs concurrently with semaphore
    tasks = [fetch_with_semaphore(url) for url in unique_urls]
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in completed:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            continue
        
        url, text = result
        results[url] = text
    
    successful = sum(1 for text in results.values() if text)
    logger.info(f"Fetched {successful}/{len(unique_urls)} pages successfully")
    
    return results


async def fetch_page_links(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    absolute: bool = True,
    internal_only: bool = False
) -> List[str]:
    """
    Extract all links from a web page.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        absolute: If True, convert relative URLs to absolute
        internal_only: If True, only return links from the same domain
        
    Returns:
        List of URLs found on the page
    """
    try:
        validate_url(url)
        
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            links = []
            base_domain = urlparse(url).netloc
            
            for link in soup.find_all("a", href=True):
                href = link["href"].strip()
                
                # Skip empty links, anchors, and javascript
                if not href or href.startswith("#") or href.startswith("javascript:"):
                    continue
                
                # Convert to absolute URL if needed
                if absolute and not href.startswith(("http://", "https://")):
                    href = urljoin(url, href)
                
                # Filter internal links if requested
                if internal_only:
                    link_domain = urlparse(href).netloc
                    if link_domain != base_domain:
                        continue
                
                links.append(href)
            
            # Remove duplicates while preserving order
            unique_links = list(dict.fromkeys(links))
            
            logger.info(f"Found {len(unique_links)} links on {url}")
            return unique_links
            
    except Exception as e:
        logger.error(f"Error extracting links from {url}: {e}")
        return []


# Synchronous wrappers for backwards compatibility
def fetch_page_text_sync(url: str, **kwargs) -> str:
    """
    Synchronous wrapper for fetch_page_text.
    
    Args:
        url: URL to fetch
        **kwargs: Additional arguments to pass to fetch_page_text
        
    Returns:
        Extracted text content
    """
    try:
        return asyncio.run(fetch_page_text(url, **kwargs))
    except Exception as e:
        logger.error(f"Error in synchronous fetch: {e}")
        return ""


def fetch_multiple_pages_sync(urls: List[str], **kwargs) -> Dict[str, str]:
    """
    Synchronous wrapper for fetch_multiple_pages.
    
    Args:
        urls: List of URLs to fetch
        **kwargs: Additional arguments to pass to fetch_multiple_pages
        
    Returns:
        Dictionary mapping URLs to their text content
    """
    try:
        return asyncio.run(fetch_multiple_pages(urls, **kwargs))
    except Exception as e:
        logger.error(f"Error in synchronous multi-fetch: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    async def main():
        # Basic usage
        text = await fetch_page_text("https://example.com")
        print(f"Fetched {len(text)} characters")
        
        # With metadata
        metadata = await fetch_page_metadata("https://example.com")
        print(f"Title: {metadata['title']}")
        
        # Fetch multiple pages
        urls = ["https://example.com", "https://python.org"]
        results = await fetch_multiple_pages(urls, max_concurrent=2)
        print(f"Fetched {len(results)} pages")
        
        # Extract links
        links = await fetch_page_links("https://example.com", internal_only=True)
        print(f"Found {len(links)} internal links")
        
        # Synchronous usage
        sync_text = fetch_page_text_sync("https://example.com")
        print(f"Sync fetch: {len(sync_text)} characters")
    
    # Run the example
    asyncio.run(main())