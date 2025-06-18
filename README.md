# Agentic Profile Authentication

A Python library for handling authentication and verification of AI agent profiles using Decentralized Identifiers (DIDs).

## Features

- DID resolution for web, HTTP, and HTTPS methods
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
    def __init__(self, store: Optional[AgenticProfileStore] = None):
        """Create a new HTTP DID resolver with optional caching."""
        pass

    async def resolve(self, did: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resolve a DID to its profile document."""
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