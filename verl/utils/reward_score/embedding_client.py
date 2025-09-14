"""
Client utilities for Text Embeddings Inference (TEI) server.

This module provides a robust client for communicating with a TEI server
hosting embedding models like Qwen3-Embedding-8B.

Features:
- Connection pooling for efficient HTTP reuse
- Retry logic with exponential backoff
- Batch request optimization
- Optional response caching
- Health check endpoints
- Graceful error handling
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache
import warnings

import numpy as np

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    warnings.warn(
        "requests library not installed. Remote embedding client will not work. "
        "Install with: pip install requests",
        RuntimeWarning
    )

try:
    import aiohttp
    import asyncio
    _HAS_ASYNC = True
except ImportError:
    _HAS_ASYNC = False

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for Text Embeddings Inference (TEI) server."""
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        connection_pool_size: int = 10,
        enable_cache: bool = True,
        cache_size: int = 1024,
    ):
        """
        Initialize the embedding client.
        
        Args:
            server_url: URL of the TEI server. Falls back to EMBEDDING_SERVER_URL env var.
            api_key: Optional API key for authentication. Falls back to EMBEDDING_SERVER_API_KEY.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_backoff_factor: Backoff factor for retries.
            connection_pool_size: Size of the HTTP connection pool.
            enable_cache: Whether to cache embedding results.
            cache_size: Maximum number of cached embeddings.
        """
        if not _HAS_REQUESTS:
            raise ImportError("requests library required for remote embedding client")
        
        # Server configuration
        self.server_url = server_url or os.getenv("EMBEDDING_SERVER_URL")
        if not self.server_url:
            raise ValueError(
                "No embedding server URL provided. Set EMBEDDING_SERVER_URL or pass server_url parameter."
            )
        
        # Clean up URL
        self.server_url = self.server_url.rstrip("/")
        
        # Authentication
        self.api_key = api_key or os.getenv("EMBEDDING_SERVER_API_KEY")
        
        # Client configuration
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        
        # Setup session with connection pooling and retries
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(
            pool_connections=connection_pool_size,
            pool_maxsize=connection_pool_size,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Cache configuration
        self.enable_cache = enable_cache
        if enable_cache:
            # Create LRU cache for embeddings
            self._get_embedding_cached = lru_cache(maxsize=cache_size)(self._get_embedding_uncached)
        
        # Server info cache
        self._server_info = None
        self._server_info_time = 0
        self._server_info_ttl = 300  # 5 minutes
        
        logger.info(f"Initialized embedding client for {self.server_url}")
    
    def health_check(self) -> bool:
        """
        Check if the embedding server is healthy.
        
        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get server information including model details.
        
        Returns:
            Server info dict or None if request fails.
        """
        # Check cache
        if self._server_info and (time.time() - self._server_info_time) < self._server_info_ttl:
            return self._server_info
        
        try:
            response = self.session.get(
                f"{self.server_url}/info",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            self._server_info = response.json()
            self._server_info_time = time.time()
            return self._server_info
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return None
    
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True,
        batch_size: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            normalize: Whether to normalize embeddings (TEI default is True).
            truncate: Whether to truncate long texts.
            batch_size: Optional batch size for processing large lists.
        
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim) or None if failed.
        """
        if not texts:
            return np.array([])
        
        # If caching is enabled and we have a single text, try cache
        if self.enable_cache and len(texts) == 1:
            cache_key = (texts[0], normalize, truncate)
            try:
                # Use the cached version
                embedding = self._get_embedding_cached(*cache_key)
                return np.array([embedding]) if embedding is not None else None
            except Exception:
                # If caching fails, continue with normal flow
                pass
        
        # For multiple texts or cache miss, make direct request
        if batch_size and len(texts) > batch_size:
            # Process in batches
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embed_batch(batch, normalize, truncate)
                if batch_embeddings is None:
                    return None
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
        else:
            # Single request
            embeddings = self._embed_batch(texts, normalize, truncate)
            return np.array(embeddings) if embeddings is not None else None
    
    def _embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True
    ) -> Optional[List[List[float]]]:
        """
        Internal method to embed a batch of texts.
        
        Returns:
            List of embedding lists or None if failed.
        """
        try:
            # Prepare request payload
            # TEI expects "inputs" field
            payload = {
                "inputs": texts,
                "normalize": normalize,
                "truncate": truncate
            }
            
            # Make request
            response = self.session.post(
                f"{self.server_url}/embed",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Parse response
            # TEI returns embeddings directly as a list of lists
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list):
                # Direct list of embeddings
                return result
            elif isinstance(result, dict):
                # May have "embeddings" key
                return result.get("embeddings", result.get("data", None))
            else:
                logger.error(f"Unexpected response format: {type(result)}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to embedding server at {self.server_url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in embed_batch: {e}")
            return None
    
    def _get_embedding_uncached(
        self,
        text: str,
        normalize: bool = True,
        truncate: bool = True
    ) -> Optional[List[float]]:
        """
        Get embedding for a single text (uncached version).
        
        Returns:
            Embedding list or None if failed.
        """
        embeddings = self._embed_batch([text], normalize, truncate)
        return embeddings[0] if embeddings else None
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncEmbeddingClient:
    """Asynchronous client for Text Embeddings Inference (TEI) server."""
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        connector_limit: int = 100,
    ):
        """
        Initialize the async embedding client.
        
        Args:
            server_url: URL of the TEI server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            connector_limit: Maximum number of connections.
        """
        if not _HAS_ASYNC:
            raise ImportError(
                "aiohttp required for async client. Install with: pip install aiohttp"
            )
        
        self.server_url = (server_url or os.getenv("EMBEDDING_SERVER_URL", "")).rstrip("/")
        if not self.server_url:
            raise ValueError("No embedding server URL provided")
        
        self.api_key = api_key or os.getenv("EMBEDDING_SERVER_API_KEY")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Connector with connection limit
        self.connector = aiohttp.TCPConnector(limit=connector_limit)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers=self.headers,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True
    ) -> Optional[np.ndarray]:
        """
        Asynchronously get embeddings for texts.
        
        Returns:
            Array of embeddings or None if failed.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.server_url}/embed",
                    json={
                        "inputs": texts,
                        "normalize": normalize,
                        "truncate": truncate
                    }
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Extract embeddings
                    if isinstance(result, list):
                        return np.array(result)
                    elif isinstance(result, dict):
                        embeddings = result.get("embeddings", result.get("data"))
                        return np.array(embeddings) if embeddings else None
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"All retry attempts failed: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None


# Convenience functions
_default_client = None

def get_default_client() -> EmbeddingClient:
    """Get or create the default embedding client."""
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


def embed_texts(
    texts: List[str],
    normalize: bool = True,
    truncate: bool = True,
    server_url: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Convenience function to embed texts using default or specified server.
    
    Args:
        texts: List of texts to embed.
        normalize: Whether to normalize embeddings.
        truncate: Whether to truncate long texts.
        server_url: Optional server URL (uses default if not specified).
    
    Returns:
        Array of embeddings or None if failed.
    """
    if server_url:
        # Create temporary client for specific server
        with EmbeddingClient(server_url=server_url) as client:
            return client.embed_texts(texts, normalize, truncate)
    else:
        # Use default client
        client = get_default_client()
        return client.embed_texts(texts, normalize, truncate)
