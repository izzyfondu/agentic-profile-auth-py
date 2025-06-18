# Agentic Profile Authentication

A Python library for handling authentication and verification of AI agent profiles using Decentralized Identifiers (DIDs).

## Features

- DID resolution for web, HTTP, and HTTPS methods
- **NEW: Built-in web DID resolver** that follows the did:web specification
- Challenge-response authentication flow
- JWT-based attestation with EdDSA signatures
- Configurable caching of resolved profiles
- Async/await support for all operations
- Type hints and Pydantic models for data validation

## Installation

```bash
pip install agentic-profile-auth
```

## Quick Start

### Basic Usage

```python
from agentic_profile_auth import HttpDidResolver, InMemoryAgenticProfileStore

# Create a resolver with caching
store = InMemoryAgenticProfileStore()
resolver = HttpDidResolver(store=store)

# Resolve a DID
async def resolve_did():
    profile, metadata = await resolver.resolve("did:web:example.com")
    print(f"Resolved profile: {profile}")
    print(f"Resolution metadata: {metadata}")

# Handle authentication
from agentic_profile_auth import handle_authorization

async def authenticate_request(auth_header: str):
    try:
        session = await handle_authorization(auth_header, store, resolver)
        print(f"Authenticated session: {session}")
    except ValueError as e:
        print(f"Authentication failed: {e}")
```

### Web DID Resolution

The library now includes a built-in web DID resolver that follows the [did:web specification](https://w3c-ccg.github.io/did-method-web/):

```python
import aiohttp
from agentic_profile_auth import (
    HttpDidResolver, 
    get_web_resolver, 
    InMemoryAgenticProfileStore
)

async def resolve_web_did():
    # Create a session for HTTP requests
    async with aiohttp.ClientSession() as session:
        
        # Create a web DID resolver
        web_resolver = get_web_resolver(session)
        
        # Create a store for caching
        store = InMemoryAgenticProfileStore()
        
        # Create the main DID resolver with web support
        resolver = HttpDidResolver(
            session=session,
            store=store,
            registry=web_resolver
        )
        
        # Resolve web DIDs
        dids = [
            "did:web:example.com",                    # https://example.com/.well-known/did.json
            "did:web:example.com:user",               # https://example.com/user/did.json
            "did:web:localhost:8080",                 # http://localhost:8080/.well-known/did.json
            "did:web:example.com:user:profile"        # https://example.com/user/profile/did.json
        ]
        
        for did in dids:
            profile, metadata = await resolver.resolve(did)
            if profile:
                print(f"✅ Resolved {did}")
            else:
                print(f"❌ Failed to resolve {did}: {metadata.get('error')}")
```

### Web DID URL Conversion

You can also convert web DIDs to their corresponding URLs:

```python
from agentic_profile_auth import web_did_to_url

# Convert web DIDs to URLs
urls = [
    web_did_to_url("did:web:example.com"),           # https://example.com/.well-known/did.json
    web_did_to_url("did:web:example.com:user"),      # https://example.com/user/did.json
    web_did_to_url("did:web:localhost:8080"),        # http://localhost:8080/.well-known/did.json
]

for url in urls:
    print(url)
```

## Authentication Flow

1. **Challenge Creation**
   - Server creates a challenge with a unique ID and secret
   - Challenge is stored in the session store
   - Challenge is sent to the client

2. **Client Authentication**
   - Client resolves the server's DID
   - Client creates a JWT with:
     - Challenge ID and secret
     - Client's DID and verification method
     - EdDSA signature using the client's private key

3. **Server Verification**
   - Server validates the JWT signature
   - Server verifies the challenge
   - Server resolves the client's DID
   - Server verifies the client's verification method

## API Reference

### DID Resolution

```python
class HttpDidResolver:
    def __init__(self, store: Optional[AgenticProfileStore] = None, registry: Optional[Dict[str, ResolverRegistry]] = None):
        """Create a new HTTP DID resolver with optional caching and method-specific resolvers."""
        pass

    async def resolve(self, did: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resolve a DID to its profile document."""
        pass

class WebDidResolver:
    """Web DID resolver that fetches DID documents from .well-known/did.json endpoints"""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """Initialize the web DID resolver."""
        pass

    async def resolve(self, did: str, parsed: ParsedDID, options: Dict[str, Any]) -> DIDResolutionResult:
        """Resolve a web DID to its DID document."""
        pass
```

### Web DID Functions

```python
def get_web_resolver(session: Optional[aiohttp.ClientSession] = None) -> Dict[str, WebDidResolver]:
    """Get a web DID resolver for use with HttpDidResolver."""
    pass

def web_did_to_url(did: str, parsed: Optional[ParsedDID] = None) -> str:
    """Convert a web DID to its corresponding URL."""
    pass

def select_protocol(path: str) -> str:
    """Select the appropriate protocol (HTTP for localhost, HTTPS for others)."""
    pass
```

### Authentication

```python
async def create_challenge(store: ClientAgentSessionStore) -> AgenticChallenge:
    """Create a new authentication challenge."""
    pass

async def handle_authorization(
    auth_header: str,
    store: ClientAgentSessionStore,
    resolver: DidResolver
) -> ClientAgentSession:
    """Handle an authorization header and return the authenticated session."""
    pass
```

### Models

- `AgenticProfile`: Represents a DID profile document
- `AgentService`: Represents an agent service in a profile
- `VerificationMethod`: Represents a verification method in a profile
- `ClientAgentSession`: Represents an authenticated session
- `AgenticChallenge`: Represents an authentication challenge
- `AgenticJwsHeader`: JWT header for agentic authentication
- `AgenticJwsPayload`: JWT payload for agentic authentication

## Web DID Specification Support

The web DID resolver implements the [did:web specification](https://w3c-ccg.github.io/did-method-web/) and supports:

- **Basic web DIDs**: `did:web:example.com` → `https://example.com/.well-known/did.json`
- **Subdomain DIDs**: `did:web:example.com:user` → `https://example.com/user/did.json`
- **Multi-level DIDs**: `did:web:example.com:user:profile` → `https://example.com/user/profile/did.json`
- **Localhost support**: `did:web:localhost:8080` → `http://localhost:8080/.well-known/did.json`
- **Query parameters**: `did:web:example.com?version=1` → `https://example.com/.well-known/did.json?version=1`

The resolver automatically:
- Selects HTTP for localhost and HTTPS for other domains
- Handles HTTP errors and network failures gracefully
- Determines content type based on `@context` presence
- Supports both `application/did+json` and `application/did+ld+json` formats

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-profile-auth.git
cd agentic-profile-auth

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[test]"
```

### Running Tests

```bash
pytest
```

### Code Style

The project uses:
- Black for code formatting
- MyPy for type checking
- Ruff for linting

## License

MIT License - see LICENSE file for details 