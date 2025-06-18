import json
import re
from typing import Optional, Dict, Any, List, Tuple, Protocol, Callable, Awaitable, TypeVar, Generic, Union
from urllib.parse import urlparse
import aiohttp
from loguru import logger
from pydantic import BaseModel

from .models import AgenticProfile, DID, VerificationMethod, Service

# Type aliases
DIDResolutionResult = Dict[str, Any]
DIDResolutionMetadata = Dict[str, Any]
DIDDocumentMetadata = Dict[str, Any]
DIDResolutionOptions = Dict[str, Any]

class ParsedDID(BaseModel):
    """Parsed DID URL components"""
    did: str
    did_url: str
    method: str
    id: str
    path: Optional[str] = None
    fragment: Optional[str] = None
    query: Optional[str] = None
    params: Optional[Dict[str, str]] = None

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

def parse_did(did_url: str) -> Optional[ParsedDID]:
    """
    Parse a DID URL into its components
    
    Args:
        did_url: The DID URL to parse
        
    Returns:
        Optional[ParsedDID]: The parsed DID components, or None if invalid
    """
    if not did_url:
        return None
        
    match = DID_MATCHER.match(did_url)
    if not match:
        return None
        
    parts = {
        'did': f"did:{match.group(1)}:{match.group(2)}",
        'method': match.group(1),
        'id': match.group(2),
        'did_url': did_url
    }
    
    if match.group(4):
        params = {}
        for param in match.group(4)[1:].split(';'):
            key, value = param.split('=')
            params[key] = value
        parts['params'] = params
        
    if match.group(6):
        parts['path'] = match.group(6)
    if match.group(7):
        parts['query'] = match.group(7)[1:]
    if match.group(8):
        parts['fragment'] = match.group(8)[1:]
        
    return ParsedDID(**parts)

class AgenticProfileStore(Protocol):
    """Protocol for storing AgenticProfiles"""
    async def save_agentic_profile(self, profile: AgenticProfile) -> None:
        """Save an AgenticProfile"""
        ...

    async def load_agentic_profile(self, did: DID) -> Optional[AgenticProfile]:
        """Load an AgenticProfile by DID"""
        ...

    async def dump(self) -> Dict[str, Any]:
        """Dump the store contents"""
        ...

class InMemoryAgenticProfileStore(AgenticProfileStore):
    """In-memory implementation of AgenticProfileStore"""
    def __init__(self):
        self._profiles: Dict[str, AgenticProfile] = {}

    async def save_agentic_profile(self, profile: AgenticProfile) -> None:
        """Save an AgenticProfile"""
        self._profiles[profile.id] = profile

    async def load_agentic_profile(self, did: DID) -> Optional[AgenticProfile]:
        """Load an AgenticProfile by DID"""
        # Remove fragment if present
        did_without_fragment = did.split('#')[0]
        return self._profiles.get(did_without_fragment)

    async def dump(self) -> Dict[str, Any]:
        """Dump the store contents"""
        return {
            "database": "None",
            "agenticProfileCache": self._profiles
        }

def as_did_resolution_result(did_document: Dict[str, Any], content_type: str = "application/json") -> DIDResolutionResult:
    """Convert a DID document to a resolution result"""
    return {
        "didDocument": did_document,
        "didDocumentMetadata": {},
        "didResolutionMetadata": {"contentType": content_type}
    }

def create_resolver_cache(store: AgenticProfileStore) -> Callable[[ParsedDID, Callable[[], Awaitable[DIDResolutionResult]]], Awaitable[DIDResolutionResult]]:
    """
    Create a resolver cache middleware
    
    Args:
        store: The AgenticProfileStore to use for caching
        
    Returns:
        A cache middleware function
    """
    async def cache_middleware(parsed: ParsedDID, resolve: Callable[[], Awaitable[DIDResolutionResult]]) -> DIDResolutionResult:
        # Check for no-cache parameter
        if parsed.params and parsed.params.get('no-cache') == 'true':
            return await resolve()  # required by DID spec
        
        # Check cache first
        profile = await store.load_agentic_profile(parsed.did)
        if profile:
            logger.debug(f"Cache hit for DID {parsed.did}")
            return as_did_resolution_result(profile.model_dump())
        
        # Resolve and cache
        result = await resolve()
        if not result.get("didResolutionMetadata", {}).get("error") and result.get("didDocument"):
            await store.save_agentic_profile(AgenticProfile(**result["didDocument"]))
        
        return result
    
    return cache_middleware

class HttpDidResolver(DidResolver):
    """
    HTTP-based DID resolver that uses the did-resolver library
    
    This resolver supports all DID methods supported by the did-resolver library.
    """
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        store: Optional[AgenticProfileStore] = None,
        registry: Optional[Dict[str, ResolverRegistry]] = None
    ):
        """
        Initialize the HTTP DID resolver
        
        Args:
            session: Optional aiohttp ClientSession to use for HTTP requests.
                    If not provided, a new session will be created.
            store: Optional AgenticProfileStore for caching resolved profiles.
                  If not provided, no caching will be used.
            registry: Optional dictionary of method-specific resolvers.
        """
        self._session = session
        self._own_session = session is None
        self._store = store
        self._registry = registry or {}
        self._cache = create_resolver_cache(store) if store else None
    
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
            profile = AgenticProfile(**result["didDocument"])
            return profile, {}
            
        except Exception as e:
            logger.exception(f"Failed to resolve DID {did}")
            return None, {
                "error": "resolutionFailed",
                "message": str(e)
            }

def create_did_resolver(
    store: Optional[AgenticProfileStore] = None,
    session: Optional[aiohttp.ClientSession] = None,
    registry: Optional[Dict[str, ResolverRegistry]] = None
) -> HttpDidResolver:
    """
    Create a DID resolver with optional caching
    
    Args:
        store: Optional AgenticProfileStore for caching
        session: Optional aiohttp ClientSession for HTTP requests
        registry: Optional dictionary of method-specific resolvers
        
    Returns:
        HttpDidResolver: A configured DID resolver
    """
    return HttpDidResolver(session=session, store=store, registry=registry) 