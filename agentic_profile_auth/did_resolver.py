import json
import re
from typing import Optional, Dict, Any, List, Tuple, Protocol, Callable, Awaitable, TypeVar, Generic, Union
from urllib.parse import urlparse
import aiohttp
from loguru import logger
from pydantic import BaseModel
import functools

from .models import AgenticProfile, DID, VerificationMethod, Service
from .did_core_types import (
    DIDResolutionResult,
    DIDResolutionMetadata,
    DIDDocumentMetadata,
    ParsedDID,
    parse_did,
    resolver_cache
)

# Type aliases
DIDResolutionOptions = Dict[str, Any]

class ResolverRegistry(Protocol):
    """Protocol for resolver registry"""
    async def resolve(self, did: str, parsed: ParsedDID, options: DIDResolutionOptions) -> DIDResolutionResult:
        """Resolve a DID"""
        ...

class DidResolver(Protocol):
    """Protocol for DID resolution"""
    async def resolve(self, did: DID) -> tuple[Optional[AgenticProfile], Dict[str, Any]]:
        """Resolve a DID to an AgenticProfile"""
        ...

# DID URL parsing regex patterns
PCT_ENCODED = r'(?:%[0-9a-fA-F]{2})'
ID_CHAR = f'(?:[a-zA-Z0-9._-]|{PCT_ENCODED})'
METHOD = r'([a-z0-9]+)'
METHOD_ID = f'((?:{ID_CHAR}*:)*({ID_CHAR}+))'
PARAM_CHAR = r'[a-zA-Z0-9_.:%-]'
PARAM = f';{PARAM_CHAR}+={PARAM_CHAR}*'
PARAMS = f'(({PARAM})*)'
PATH = r'(/[^#?]*)?'
QUERY = r'([?][^#]*)?'
FRAGMENT = r'(#.*)?'
DID_MATCHER = re.compile(f'^did:{METHOD}:{METHOD_ID}{PARAMS}{PATH}{QUERY}{FRAGMENT}$')

def as_did_resolution_result(did_document: Dict[str, Any], content_type: str = "application/json") -> DIDResolutionResult:
    """Convert a DID document to a resolution result"""
    return {
        "didDocument": did_document,
        "didDocumentMetadata": {},
        "didResolutionMetadata": {"contentType": content_type}
    }

class HttpDidResolver(DidResolver):
    """
    HTTP-based DID resolver that uses the did-resolver library
    
    This resolver supports all DID methods supported by the did-resolver library.
    """
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        store: Optional[Any] = None,
        registry: Optional[Dict[str, ResolverRegistry]] = None
    ):
        """
        Initialize the HTTP DID resolver
        
        Args:
            session: Optional aiohttp ClientSession to use for HTTP requests.
                    If not provided, a new session will be created.
            store: Optional store for caching resolved profiles.
                  If not provided, no caching will be used.
            registry: Optional dictionary of method-specific resolvers.
        """
        self._session = session
        self._own_session = session is None
        self._store = store
        self._registry = registry or {}
        self._cache = functools.partial(resolver_cache, store) if store else None
    
    async def __aenter__(self):
        """Create a new session if one wasn't provided"""
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the session if we created it"""
        if self._own_session and self._session:
            await self._session.close()
            self._session = None
    
    async def resolve(self, did: DID) -> tuple[Optional[AgenticProfile], Dict[str, Any]]:
        """
        Resolve a DID to an AgenticProfile
        
        Args:
            did: The DID to resolve
            
        Returns:
            tuple[Optional[AgenticProfile], Dict[str, Any]]: The resolved profile and metadata
        """
        try:
            # Parse DID
            parsed = parse_did(did)
            if not parsed:
                return None, {
                    "error": "invalidDid",
                    "message": f"Invalid DID format: {did}"
                }
            
            # Get resolver for method
            resolver = self._registry.get(parsed.method)
            if not resolver:
                return None, {
                    "error": "unsupportedDidMethod",
                    "message": f"Unsupported DID method: {parsed.method}"
                }
            
            # Resolve DID
            async def resolve_did() -> DIDResolutionResult:
                return await resolver.resolve(did, parsed, {})
            
            # Use cache if available
            if self._cache:
                result = await self._cache(parsed, resolve_did)
            else:
                result = await resolve_did()
            
            if result.get("didResolutionMetadata", {}).get("error"):
                return None, result["didResolutionMetadata"]
            
            if not result.get("didDocument"):
                return None, {
                    "error": "notFound",
                    "message": f"No DID document found for {did}"
                }
            
            # Convert DIDDocument to AgenticProfile
            did_document = result["didDocument"]
            if isinstance(did_document, AgenticProfile):
                profile = did_document
            else:
                profile = AgenticProfile(**did_document)
            return profile, {}
            
        except Exception as e:
            logger.exception(f"Failed to resolve DID {did}")
            return None, {
                "error": "resolutionFailed",
                "message": str(e)
            }

def create_did_resolver(
    store: Optional[Any] = None,
    session: Optional[aiohttp.ClientSession] = None,
    registry: Optional[Dict[str, ResolverRegistry]] = None
) -> HttpDidResolver:
    """
    Create a DID resolver with optional caching
    
    Args:
        store: Optional store for caching
        session: Optional aiohttp ClientSession for HTTP requests
        registry: Optional dictionary of method-specific resolvers
        
    Returns:
        HttpDidResolver: A configured DID resolver
    """
    return HttpDidResolver(session=session, store=store, registry=registry) 