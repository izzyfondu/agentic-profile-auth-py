import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from agentic_profile_auth.server_authentication import (
    create_challenge,
    handle_authorization,
    validate_auth_token,
    resolve_verification_method,
    ClientAgentSession,
    ClientAgentSessionStore,
    AgenticChallenge,
    AgenticJwsHeader,
    AgenticJwsPayload,
    DID,
    FragmentID,
    AgenticProfile,
    DidResolver
)
from agentic_profile_auth.models import (
    AgenticProfile,
    AgentService,
    VerificationMethod
)

# Test data
SAMPLE_DID = "did:web:example.com"
SAMPLE_FRAGMENT = "#key-1"
SAMPLE_VERIFICATION_ID = f"{SAMPLE_DID}{SAMPLE_FRAGMENT}"
SAMPLE_CHALLENGE_ID = "challenge-123"
SAMPLE_CHALLENGE_SECRET = "secret-456"
SAMPLE_AUTH_TOKEN = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJjaGFsbGVuZ2UiOnsiaWQiOiJjaGFsbGVuZ2UtMTIzIiwic2VjcmV0Ijoic2VjcmV0LTQ1NiJ9LCJhdHRlc3QiOnsiYWdlbnREaWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tIiwidmVyaWZpY2F0aW9uSWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tI2tleS0xIn19.signature"

class MockStore(ClientAgentSessionStore):
    """Mock implementation of ClientAgentSessionStore"""
    def __init__(self):
        self.sessions = {}
        self.next_id = 1

    async def create_client_agent_session(self, secret: str) -> str:
        session_id = f"session-{self.next_id}"
        self.next_id += 1
        self.sessions[session_id] = ClientAgentSession(
            challenge_id=session_id,
            challenge=secret
        )
        return session_id

    async def fetch_client_agent_session(self, challenge_id: str) -> ClientAgentSession:
        return self.sessions.get(challenge_id)

    async def update_client_agent_session(self, challenge_id: str, updates: dict) -> None:
        if challenge_id in self.sessions:
            session = self.sessions[challenge_id]
            for key, value in updates.items():
                setattr(session, key, value)

class MockDidResolver:
    """Mock implementation of DidResolver"""
    def __init__(self, profile=None):
        self.profile = profile

    async def resolve(self, did: DID) -> tuple[dict, dict]:
        if self.profile:
            return self.profile, {}
        return None, {"error": "notFound"}

@pytest.fixture
def store():
    """Create a mock store"""
    return MockStore()

@pytest.fixture
def did_resolver():
    """Create a mock DID resolver"""
    return MockDidResolver()

@pytest.mark.asyncio
async def test_create_challenge(store):
    """Test challenge creation"""
    challenge = await create_challenge(store)
    
    assert isinstance(challenge, AgenticChallenge)
    assert challenge.type == "agentic-challenge"
    assert "id" in challenge.challenge
    assert "secret" in challenge.challenge
    
    # Verify session was created
    session = await store.fetch_client_agent_session(challenge.challenge["id"])
    assert session is not None
    assert session.challenge == challenge.challenge["secret"]

@pytest.mark.asyncio
async def test_handle_authorization_invalid_type(store, did_resolver):
    """Test handling invalid authorization type"""
    with pytest.raises(ValueError, match="Unsupported authorization type"):
        await handle_authorization("Bearer token", store, did_resolver)

@pytest.mark.asyncio
async def test_handle_authorization_missing_token(store, did_resolver):
    """Test handling missing authorization token"""
    with pytest.raises(ValueError, match="Missing Agentic authorization token"):
        await handle_authorization("Agentic", store, did_resolver)

@pytest.mark.asyncio
async def test_handle_authorization_invalid_token(store, did_resolver):
    """Test handling invalid authorization token"""
    with pytest.raises(ValueError, match="Failed to parse agentic token"):
        await handle_authorization("Agentic invalid.token", store, did_resolver)

@pytest.mark.asyncio
async def test_handle_authorization_missing_challenge(store, did_resolver):
    """Test handling missing challenge ID"""
    with pytest.raises(ValueError, match="Agent token missing payload.challenge.id"):
        await handle_authorization(
            "Agentic eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhdHRlc3QiOnsiYWdlbnREaWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tIn19.signature",
            store,
            did_resolver
        )

@pytest.mark.asyncio
async def test_handle_authorization_invalid_session(store, did_resolver):
    """Test handling invalid session"""
    result = await handle_authorization(
        f"Agentic {SAMPLE_AUTH_TOKEN}",
        store,
        did_resolver
    )
    assert result is None

@pytest.mark.asyncio
async def test_validate_auth_token_invalid_alg(store, did_resolver):
    """Test validating token with invalid algorithm"""
    session = await store.create_client_agent_session(SAMPLE_CHALLENGE_SECRET)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjaGFsbGVuZ2UiOnsiaWQiOiJjaGFsbGVuZ2UtMTIzIiwic2VjcmV0Ijoic2VjcmV0LTQ1NiJ9LCJhdHRlc3QiOnsiYWdlbnREaWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tIiwidmVyaWZpY2F0aW9uSWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tI2tleS0xIn19.signature"
    
    with pytest.raises(ValueError, match="Only EdDSA JWS is currently supported"):
        await validate_auth_token(token, session, store, did_resolver)

@pytest.mark.asyncio
async def test_validate_auth_token_missing_challenge(store, did_resolver):
    """Test validating token with missing challenge"""
    session = await store.create_client_agent_session(SAMPLE_CHALLENGE_SECRET)
    token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhdHRlc3QiOnsiYWdlbnREaWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tIiwidmVyaWZpY2F0aW9uSWQiOiJkaWQ6d2ViOmV4YW1wbGUuY29tI2tleS0xIn19.signature"
    
    with pytest.raises(ValueError, match="Missing 'challenge' from agentic JWS payload"):
        await validate_auth_token(token, session, store, did_resolver)

@pytest.mark.asyncio
async def test_validate_auth_token_missing_attest(store, did_resolver):
    """Test validating token with missing attest"""
    session = await store.create_client_agent_session(SAMPLE_CHALLENGE_SECRET)
    token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJjaGFsbGVuZ2UiOnsiaWQiOiJjaGFsbGVuZ2UtMTIzIiwic2VjcmV0Ijoic2VjcmV0LTQ1NiJ9fQ.signature"
    
    with pytest.raises(ValueError, match="Missing 'attest' from agentic JWS payload"):
        await validate_auth_token(token, session, store, did_resolver)

@pytest.mark.asyncio
async def test_resolve_verification_method_not_found(did_resolver):
    """Test resolving non-existent verification method"""
    profile = AgenticProfile(
        id=SAMPLE_DID,
        name="Test Agent",
        service=[
            AgentService(
                id=SAMPLE_DID,
                type="AgentService",
                service_endpoint="https://example.com",
                name="Test Service",
                capability_invocation=[]
            )
        ]
    )
    
    with pytest.raises(ValueError, match="Verification id does not match any entries in the agents capabilityInvocation list"):
        await resolve_verification_method(profile, SAMPLE_DID, SAMPLE_VERIFICATION_ID, did_resolver)

@pytest.mark.asyncio
async def test_resolve_verification_method_invalid_type(did_resolver):
    """Test resolving verification method with invalid type"""
    profile = AgenticProfile(
        id=SAMPLE_DID,
        name="Test Agent",
        service=[
            AgentService(
                id=SAMPLE_DID,
                type="AgentService",
                service_endpoint="https://example.com",
                name="Test Service",
                capability_invocation=[
                    VerificationMethod(
                        id=SAMPLE_VERIFICATION_ID,
                        type="InvalidType",
                        controller=SAMPLE_DID
                    )
                ]
            )
        ]
    )
    
    with pytest.raises(ValueError, match="Verification id does not match any entries in the agents capabilityInvocation list"):
        await resolve_verification_method(profile, SAMPLE_DID, SAMPLE_VERIFICATION_ID, did_resolver) 