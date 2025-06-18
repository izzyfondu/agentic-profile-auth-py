import os
from typing import Optional, Protocol, Dict, Any
from loguru import logger
from jose import jwt

from .models import (
    AgenticProfile,
    AgenticChallenge,
    AgenticJwsHeader,
    AgenticJwsPayload,
    ClientAgentSession,
    ClientAgentSessionUpdates,
    DID,
    FragmentID,
    VerificationMethod
)
from .b64u import base64url_to_object, object_to_base64url, bytes_to_base64url
from .ed25519 import verify

# Constants
AGENTIC_CHALLENGE_TYPE = "agentic-challenge"

class ClientAgentSessionStore(Protocol):
    """Protocol for client agent session storage"""
    async def create_client_agent_session(self, secret: str) -> str:
        """Create a new client agent session and return its ID"""
        ...

    async def fetch_client_agent_session(self, challenge_id: str) -> Optional[ClientAgentSession]:
        """Fetch a client agent session by challenge ID"""
        ...

    async def update_client_agent_session(self, challenge_id: str, updates: ClientAgentSessionUpdates) -> None:
        """Update a client agent session"""
        ...

class DidResolver(Protocol):
    """Protocol for DID resolution"""
    async def resolve(self, did: DID) -> tuple[Optional[AgenticProfile], Dict[str, Any]]:
        """Resolve a DID to an AgenticProfile"""
        ...

def unpack_compact_jws(jws: str) -> tuple[AgenticJwsHeader, AgenticJwsPayload, str]:
    """Unpack a compact JWS into its components"""
    b64u_header, b64u_payload, b64u_signature = jws.split('.')
    header = base64url_to_object(b64u_header)
    payload = base64url_to_object(b64u_payload)
    return header, payload, b64u_signature

async def create_challenge(store: ClientAgentSessionStore) -> AgenticChallenge:
    """Create a new authentication challenge"""
    secret = bytes_to_base64url(os.urandom(32))
    challenge_id = await store.create_client_agent_session(secret)
    return AgenticChallenge(
        type=AGENTIC_CHALLENGE_TYPE,
        challenge={"id": challenge_id, "secret": secret}
    )

async def handle_authorization(
    authorization: str,
    store: ClientAgentSessionStore,
    did_resolver: DidResolver
) -> Optional[ClientAgentSession]:
    """
    Handle an HTTP authorization header
    
    Args:
        authorization: The authorization header value, of the form "Agentic <JWT>"
        store: The client agent session store
        did_resolver: The DID resolver
        
    Returns:
        Optional[ClientAgentSession]: The client agent session if valid, None if challenge ID was found but is now invalid
        
    Raises:
        ValueError: If authorization header is invalid
    """
    tokens = authorization.strip().split()
    if tokens[0].lower() != "agentic":
        raise ValueError(f"Unsupported authorization type: {tokens[0]}")
    if len(tokens) < 2:
        raise ValueError("Missing Agentic authorization token")
    
    auth_token = tokens[1]
    
    try:
        header, payload = unpack_compact_jws(auth_token)[:2]
    except Exception as e:
        raise ValueError(f"Failed to parse agentic token: {str(e)} token: {auth_token}")
    
    challenge_id = payload.get("challenge", {}).get("id")
    if not challenge_id:
        raise ValueError("Agent token missing payload.challenge.id")
    
    session = await store.fetch_client_agent_session(challenge_id)
    if not session:
        logger.warning(f"Failed to find agent session {challenge_id}")
        return None
    
    if not session.auth_token:
        # Session has not started yet, so validate auth token
        return await validate_auth_token(auth_token, session, store, did_resolver)
    
    if session.auth_token != auth_token:
        raise ValueError("Incorrect authorization token; Does not match one used for validation")
    
    return session

async def validate_auth_token(
    auth_token: str,
    session: ClientAgentSession,
    store: ClientAgentSessionStore,
    did_resolver: DidResolver
) -> ClientAgentSession:
    """
    Validate an authentication token
    
    Args:
        auth_token: The authentication token to validate
        session: The client agent session
        store: The client agent session store
        did_resolver: The DID resolver
        
    Returns:
        ClientAgentSession: The updated client agent session
        
    Raises:
        ValueError: If token validation fails
    """
    header, payload = unpack_compact_jws(auth_token)[:2]
    
    if header.get("alg") != "EdDSA":
        raise ValueError("Only EdDSA JWS is currently supported")
    
    challenge = payload.get("challenge")
    attest = payload.get("attest")
    
    if not challenge:
        raise ValueError("Missing 'challenge' from agentic JWS payload")
    if not attest:
        raise ValueError("Missing 'attest' from agentic JWS payload")
    
    agent_did = attest.get("agentDid")
    verification_id = attest.get("verificationId")
    
    if not agent_did:
        raise ValueError("Missing 'attest.agentDid' from agentic JWS payload")
    if not verification_id:
        raise ValueError("Missing 'attest.verificationId' from agentic JWS payload")
    
    expected_challenge = session.challenge
    signed_challenge = challenge.get("secret")
    
    if expected_challenge != signed_challenge:
        raise ValueError(f"Signed challenge is different than expected: {signed_challenge} != {expected_challenge}")
    
    # Verify publicKey in signature is from user specified in agentDid
    profile, resolution_metadata = await did_resolver.resolve(agent_did)
    if resolution_metadata.get("error"):
        raise ValueError(f"Failed to resolve agentic profile from DID: {resolution_metadata['error']}")
    if not profile:
        raise ValueError("DID resolver failed to return agentic profile")
    
    verification_method = await resolve_verification_method(profile, agent_did, verification_id, did_resolver)
    if verification_method.type != "JsonWebKey2020":
        raise ValueError("Unsupported verification type, please use JsonWebKey2020 for agents")
    
    public_key_jwk = verification_method.public_key_jwk
    if not public_key_jwk:
        raise ValueError("Missing 'publicKeyJwk' property in verification method")
    
    kty = public_key_jwk.get("kty")
    alg = public_key_jwk.get("alg")
    crv = public_key_jwk.get("crv")
    b64u_public_key = public_key_jwk.get("x")
    
    if kty != "OKP":
        raise ValueError("JWK kty must be OKP")
    if alg != "EdDSA":
        raise ValueError("JWK alg must be EdDSA")
    if crv != "Ed25519":
        raise ValueError("JWK crv must be Ed25519")
    if not b64u_public_key:
        raise ValueError("JWK must provide 'x' as the public key")
    
    b64u_header, b64u_payload, b64u_signature = auth_token.split('.')
    message = f"{b64u_header}.{b64u_payload}"
    
    is_valid = await verify(b64u_signature, message, b64u_public_key)
    if not is_valid:
        raise ValueError("Invalid signed challenge and attestation")
    
    session_updates = ClientAgentSessionUpdates(agent_did=agent_did, auth_token=auth_token)
    await store.update_client_agent_session(challenge["id"], session_updates)
    
    return ClientAgentSession(**{**session.dict(), **session_updates.dict()})

async def resolve_verification_method(
    profile: AgenticProfile,
    agent_did: DID,
    verification_id: FragmentID,
    did_resolver: DidResolver
) -> VerificationMethod:
    """
    Resolve a verification method from a DID document
    
    Args:
        profile: The agentic profile
        agent_did: The agent DID
        verification_id: The verification method ID
        did_resolver: The DID resolver
        
    Returns:
        VerificationMethod: The resolved verification method
        
    Raises:
        ValueError: If verification method cannot be resolved
    """
    # Find agent
    agent = next((s for s in profile.service or [] if s.id == agent_did), None)
    if not agent:
        raise ValueError(f"Failed to find agent service for {agent_did}")
    
    # Does this agent have the indicated verification?
    method_or_id = next((m for m in agent.capability_invocation if m == verification_id), None)
    if not method_or_id:
        raise ValueError(f"Verification id does not match any entries in the agents capabilityInvocation list: {verification_id}")
    
    if isinstance(method_or_id, VerificationMethod):
        return method_or_id
    
    if not isinstance(method_or_id, str):
        raise ValueError(f"Unexpected capabilityInvocation type: {method_or_id}")
    
    # Is this verification method in another did document/agentic profile?
    linked_did = method_or_id.split("#")[0]
    if profile.id != linked_did:
        logger.debug(f"Redirecting to linked agentic profile to resolve verification method {linked_did}")
        profile, resolution_metadata = await did_resolver.resolve(linked_did)
        if resolution_metadata.get("error"):
            raise ValueError(f"Failed to resolve agentic profile from DID: {resolution_metadata['error']}")
        if not profile:
            raise ValueError("DID resolver failed to return agentic profile")
    
    verification_method = next((m for m in profile.verification_method or [] if m.id == verification_id), None)
    if not verification_method:
        raise ValueError(f"Verification id does not match any listed verification methods: {verification_id}")
    
    return verification_method 