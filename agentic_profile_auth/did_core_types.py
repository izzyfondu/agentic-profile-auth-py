# Copyright 2018 Consensys AG
# Licensed under the Apache License, Version 2.0
# See: http://www.apache.org/licenses/LICENSE-2.0

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
import re
from loguru import logger

Extensible = Dict[str, Any]

@dataclass
class DIDResolutionMetadata:
    contentType: Optional[str] = None
    error: Optional[str] = None
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DIDDocumentMetadata:
    created: Optional[str] = None
    updated: Optional[str] = None
    deactivated: Optional[bool] = None
    versionId: Optional[str] = None
    nextUpdate: Optional[str] = None
    nextVersionId: Optional[str] = None
    equivalentId: Optional[str] = None
    canonicalId: Optional[str] = None
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

KeyCapabilitySection = [
    'authentication',
    'assertionMethod',
    'keyAgreement',
    'capabilityInvocation',
    'capabilityDelegation',
]

@dataclass
class JsonWebKey:
    alg: Optional[str] = None
    crv: Optional[str] = None
    e: Optional[str] = None
    ext: Optional[bool] = None
    key_ops: Optional[List[str]] = None
    kid: Optional[str] = None
    kty: Optional[str] = None
    n: Optional[str] = None
    use: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationMethod:
    id: str
    type: str
    controller: str
    publicKeyBase58: Optional[str] = None
    publicKeyBase64: Optional[str] = None
    publicKeyJwk: Optional[JsonWebKey] = None
    publicKeyHex: Optional[str] = None
    publicKeyMultibase: Optional[str] = None
    blockchainAccountId: Optional[str] = None
    ethereumAddress: Optional[str] = None
    conditionOr: Optional[List['VerificationMethod']] = None
    conditionAnd: Optional[List['VerificationMethod']] = None
    threshold: Optional[int] = None
    conditionThreshold: Optional[List['VerificationMethod']] = None
    conditionWeightedThreshold: Optional[List[Dict[str, Any]]] = None
    conditionDelegated: Optional[str] = None
    relationshipParent: Optional[List[str]] = None
    relationshipChild: Optional[List[str]] = None
    relationshipSibling: Optional[List[str]] = None
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Service:
    id: str
    type: str
    serviceEndpoint: Union[str, Dict[str, Any], List[Any]]
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DIDDocument:
    id: str
    context: Optional[Union[str, List[str]]] = None
    alsoKnownAs: Optional[List[str]] = None
    controller: Optional[Union[str, List[str]]] = None
    verificationMethod: Optional[List[VerificationMethod]] = None
    service: Optional[List[Service]] = None
    publicKey: Optional[List[VerificationMethod]] = None  # Deprecated
    authentication: Optional[List[Union[str, VerificationMethod]]] = None
    assertionMethod: Optional[List[Union[str, VerificationMethod]]] = None
    keyAgreement: Optional[List[Union[str, VerificationMethod]]] = None
    capabilityInvocation: Optional[List[Union[str, VerificationMethod]]] = None
    capabilityDelegation: Optional[List[Union[str, VerificationMethod]]] = None
    # Extensible
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DIDResolutionResult:
    didResolutionMetadata: DIDResolutionMetadata
    didDocument: Optional[DIDDocument]
    didDocumentMetadata: DIDDocumentMetadata
    context: Optional[Union[str, List[str]]] = None

@dataclass
class ParsedDID:
    did: str
    did_url: str
    method: str
    id: str
    path: Optional[str] = None
    fragment: Optional[str] = None
    query: Optional[str] = None
    params: Optional[Dict[str, str]] = None

# DID URL parsing regex
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
            if '=' in param:
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

# In-memory cache for DID resolution
class InMemoryDIDCache:
    def __init__(self):
        self.cache: Dict[str, DIDResolutionResult] = {}

    async def __call__(self, parsed: ParsedDID, resolve: Callable[[], Awaitable[DIDResolutionResult]]) -> DIDResolutionResult:
        if parsed.params and parsed.params.get('no-cache') == 'true':
            return await resolve()
        cached = self.cache.get(parsed.did_url)
        if cached is not None:
            return cached
        result = await resolve()
        if not getattr(result.didResolutionMetadata, 'error', None) == 'notFound':
            self.cache[parsed.did_url] = result
        return result

def as_did_resolution_result(did_document, content_type="application/json"):
    return {
        "didDocument": did_document,
        "didDocumentMetadata": {},
        "didResolutionMetadata": {"contentType": content_type}
    }

async def resolver_cache(store, parsed: ParsedDID, resolve: Callable[[], Awaitable[DIDResolutionResult]]) -> DIDResolutionResult:
    if parsed.params and parsed.params.get('no-cache') == 'true':
        logger.debug(f"Bypassing cache for {parsed.did}")
        return await resolve()
    profile = await store.load_agentic_profile(parsed.did)
    if profile:
        logger.debug(f"Cache hit for {parsed.did}")
        # Convert AgenticProfile to dict if needed
        if hasattr(profile, 'model_dump'):
            profile_dict = profile.model_dump()
        else:
            profile_dict = profile
        return as_did_resolution_result(profile_dict)
    logger.debug(f"Cache miss for {parsed.did}, resolving...")
    result = await resolve()
    # Check if resolution was successful (no error and has document)
    resolution_metadata = result.get('didResolutionMetadata', {})
    has_error = resolution_metadata.get('error') is not None
    has_document = result.get('didDocument') is not None
    
    if not has_error and has_document:
        # Convert dict to AgenticProfile if needed
        did_document = result['didDocument']
        if isinstance(did_document, dict):
            from .models import AgenticProfile
            profile = AgenticProfile(**did_document)
        else:
            profile = did_document
        logger.debug(f"Saving profile for {parsed.did}")
        await store.save_agentic_profile(profile)
    else:
        logger.debug(f"Not saving profile for {parsed.did}, error or missing document.")
    return result 