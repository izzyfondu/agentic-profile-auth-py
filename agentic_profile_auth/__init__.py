from .models import (
    AgenticProfile,
    AgentService,
    VerificationMethod,
    AgenticChallenge,
    AgenticJwsHeader,
    AgenticJwsPayload,
    ClientAgentSession,
    ClientAgentSessionUpdates,
    DID,
    FragmentID,
    UserID,
    InMemoryAgenticProfileStore
)

from .ed25519 import verify, create_key_pair
from .b64u import (
    base64_to_base64url,
    base64url_to_base64,
    base64url_to_bytes,
    bytes_to_base64url,
    base64url_to_object,
    object_to_base64url
)

from .server_authentication import (
    ClientAgentSessionStore,
    DidResolver,
    create_challenge,
    handle_authorization,
    validate_auth_token,
    resolve_verification_method,
    AGENTIC_CHALLENGE_TYPE
)

from .web_did_resolver import (
    WebDidResolver,
    get_web_resolver,
    web_did_to_url,
    select_protocol
)

from .did_resolver import (
    HttpDidResolver,
    create_did_resolver,
    ResolverRegistry
)

__version__ = "0.6.0"

__all__ = [
    # Models
    "AgenticProfile",
    "AgentService",
    "VerificationMethod",
    "AgenticChallenge",
    "AgenticJwsHeader",
    "AgenticJwsPayload",
    "ClientAgentSession",
    "ClientAgentSessionUpdates",
    "DID",
    "FragmentID",
    "UserID",
    "InMemoryAgenticProfileStore",
    
    # Protocols
    "ClientAgentSessionStore",
    "DidResolver",
    "ResolverRegistry",
    
    # Resolvers
    "HttpDidResolver",
    "WebDidResolver",
    "create_did_resolver",
    "get_web_resolver",
    
    # Functions
    "verify",
    "create_key_pair",
    "create_challenge",
    "handle_authorization",
    "validate_auth_token",
    "resolve_verification_method",
    "web_did_to_url",
    "select_protocol",
    
    # Utilities
    "base64_to_base64url",
    "base64url_to_base64",
    "base64url_to_bytes",
    "bytes_to_base64url",
    "base64url_to_object",
    "object_to_base64url",
    
    # Constants
    "AGENTIC_CHALLENGE_TYPE"
] 