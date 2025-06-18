import json
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import aiohttp
from loguru import logger

from .did_core_types import (
    DIDResolutionResult,
    DIDResolutionMetadata,
    DIDDocumentMetadata,
    ParsedDID,
    as_did_resolution_result
)

def select_protocol(path: str) -> str:
    """Select the appropriate protocol based on the domain"""
    return "http" if path.startswith("localhost") else "https"

def web_did_to_url(did: str, parsed: Optional[ParsedDID] = None) -> str:
    """
    Convert a web DID to its corresponding URL
    
    Args:
        did: The web DID to convert
        parsed: Optional parsed DID object. If not provided, will be parsed from the DID string.
        
    Returns:
        str: The URL where the DID document can be fetched
        
    Raises:
        ValueError: If the DID is not a valid web DID or cannot be parsed
    """
    if not parsed:
        from .did_core_types import parse_did
        parsed = parse_did(did)
    
    if not parsed:
        raise ValueError(f"Failed to parse DID: {did}")
    
    if parsed.method != 'web':
        raise ValueError("Only did:web is supported")
    
    if parsed.path:
        raise ValueError("Web DIDs do not support paths")
    
    id_parts = parsed.id.split(":")
    
    # Special handling for localhost:port format
    if len(id_parts) >= 2 and id_parts[0] == "localhost":
        domain = f"{id_parts[0]}:{id_parts[1]}"
        remaining_parts = id_parts[2:]
        if not remaining_parts:
            # Single domain:port, use .well-known
            path = f"{domain}/.well-known/did.json"
        else:
            # Sub-paths, use /did.json
            path = "/".join([domain] + remaining_parts) + "/did.json"
    else:
        if len(id_parts) == 1:
            path = f"{id_parts[0]}/.well-known/did.json"
        else:
            path = "/".join(id_parts) + "/did.json"
    
    if parsed.query:
        path += '?' + parsed.query
    
    protocol = select_protocol(id_parts[0])
    return f"{protocol}://{path}"

class WebDidResolver:
    """
    Web DID resolver that fetches DID documents from .well-known/did.json endpoints
    
    This resolver follows the did:web specification and handles:
    - did:web:example.com -> https://example.com/.well-known/did.json
    - did:web:example.com:user -> https://example.com/user/did.json
    - did:web:localhost:8080 -> http://localhost:8080/.well-known/did.json
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize the web DID resolver
        
        Args:
            session: Optional aiohttp ClientSession to use for HTTP requests.
                    If not provided, a new session will be created for each request.
        """
        self._session = session
        self._own_session = session is None
    
    async def resolve(
        self, 
        did: str, 
        parsed: ParsedDID, 
        options: Dict[str, Any]
    ) -> DIDResolutionResult:
        """
        Resolve a web DID to its DID document
        
        Args:
            did: The web DID to resolve
            parsed: The parsed DID object
            options: Resolution options (unused for web DIDs)
            
        Returns:
            DIDResolutionResult: The resolution result containing the DID document
        """
        url = None
        try:
            url = web_did_to_url(did, parsed)
            
            # Use provided session or create a temporary one
            if self._session:
                response = await self._session.get(url)
            else:
                async with aiohttp.ClientSession() as temp_session:
                    response = await temp_session.get(url)
            
            if not response.ok:
                raise Exception(f"HTTP error {response.status}")
            
            did_document = await response.json()
            
            if not did_document or not isinstance(did_document, dict):
                raise Exception("Invalid DID Document format")
            
            # Determine content type based on @context presence
            content_type = "application/did+ld+json" if did_document.get("@context") else "application/did+json"
            
            return {
                "didDocument": did_document,
                "didDocumentMetadata": {},
                "didResolutionMetadata": {"contentType": content_type}
            }
            
        except Exception as error:
            logger.warning(f"Failed to resolve web DID {did}: {str(error)}")
            return {
                "didDocument": None,
                "didDocumentMetadata": {},
                "didResolutionMetadata": {
                    "error": f"Failed to resolve DID document{url and f' from {url}' or ''}",
                    "message": str(error)
                }
            }

def get_web_resolver(session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
    """
    Get a web DID resolver for use with the HttpDidResolver
    
    Args:
        session: Optional aiohttp ClientSession to use for HTTP requests
        
    Returns:
        Dict[str, Any]: A dictionary mapping 'web' to the web DID resolver
    """
    return {"web": WebDidResolver(session)}