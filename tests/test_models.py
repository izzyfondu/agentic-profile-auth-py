import pytest
from datetime import datetime

from agentic_profile_auth.models import (
    AgenticProfile,
    AgentService,
    VerificationMethod,
    AgenticChallenge,
    AgenticJwsHeader,
    AgenticJwsPayload,
    ClientAgentSession,
    ClientAgentSessionUpdates,
    DID,
    FragmentID
)

# Test data
SAMPLE_DID = "did:web:example.com"
SAMPLE_FRAGMENT = "#key-1"
SAMPLE_VERIFICATION_ID = f"{SAMPLE_DID}{SAMPLE_FRAGMENT}"
SAMPLE_CHALLENGE_ID = "challenge-123"
SAMPLE_CHALLENGE_SECRET = "secret-456"

def test_agentic_profile_creation():
    """Test AgenticProfile creation"""
    profile = AgenticProfile(
        id=SAMPLE_DID,
        name="Test Profile",
        verification_method=[],
        service=[],
        ttl=3600
    )
    
    assert profile.id == SAMPLE_DID
    assert profile.name == "Test Profile"
    assert profile.verification_method == []
    assert profile.service == []
    assert profile.ttl == 3600

def test_agentic_profile_default_ttl():
    """Test AgenticProfile default TTL"""
    profile = AgenticProfile(
        id=SAMPLE_DID,
        name="Test Profile"
    )
    
    assert profile.ttl == 86400  # Default TTL is one day

def test_agent_service_creation():
    """Test AgentService creation"""
    service = AgentService(
        id=SAMPLE_DID,
        type="AgentService",
        service_endpoint="https://example.com/agent",
        name="Test Agent",
        capability_invocation=[]
    )
    
    assert service.id == SAMPLE_DID
    assert service.type == "AgentService"
    assert service.service_endpoint == "https://example.com/agent"
    assert service.name == "Test Agent"
    assert service.capability_invocation == []

def test_verification_method_creation():
    """Test VerificationMethod creation"""
    method = VerificationMethod(
        id=SAMPLE_VERIFICATION_ID,
        type="JsonWebKey2020",
        controller=SAMPLE_DID,
        public_key_jwk={
            "kty": "OKP",
            "alg": "EdDSA",
            "crv": "Ed25519",
            "x": "base64url-encoded-public-key"
        }
    )
    
    assert method.id == SAMPLE_VERIFICATION_ID
    assert method.type == "JsonWebKey2020"
    assert method.controller == SAMPLE_DID
    assert method.public_key_jwk["kty"] == "OKP"
    assert method.public_key_jwk["alg"] == "EdDSA"
    assert method.public_key_jwk["crv"] == "Ed25519"

def test_agentic_challenge_creation():
    """Test AgenticChallenge creation"""
    challenge = AgenticChallenge(
        type="agentic-challenge",
        challenge={
            "id": SAMPLE_CHALLENGE_ID,
            "secret": SAMPLE_CHALLENGE_SECRET
        }
    )
    
    assert challenge.type == "agentic-challenge"
    assert challenge.challenge["id"] == SAMPLE_CHALLENGE_ID
    assert challenge.challenge["secret"] == SAMPLE_CHALLENGE_SECRET

def test_agentic_jws_header_creation():
    """Test AgenticJwsHeader creation"""
    header = AgenticJwsHeader()
    
    assert header.alg == "EdDSA"
    assert header.typ == "JWT"

def test_agentic_jws_payload_creation():
    """Test AgenticJwsPayload creation"""
    payload = AgenticJwsPayload(
        challenge={
            "id": SAMPLE_CHALLENGE_ID,
            "secret": SAMPLE_CHALLENGE_SECRET
        },
        attest={
            "agentDid": SAMPLE_DID,
            "verificationId": SAMPLE_VERIFICATION_ID
        }
    )
    
    assert payload.challenge["id"] == SAMPLE_CHALLENGE_ID
    assert payload.challenge["secret"] == SAMPLE_CHALLENGE_SECRET
    assert payload.attest["agentDid"] == SAMPLE_DID
    assert payload.attest["verificationId"] == SAMPLE_VERIFICATION_ID

def test_client_agent_session_creation():
    """Test ClientAgentSession creation"""
    now = datetime.utcnow()
    session = ClientAgentSession(
        challenge_id=SAMPLE_CHALLENGE_ID,
        challenge=SAMPLE_CHALLENGE_SECRET,
        agent_did=SAMPLE_DID,
        auth_token="token",
        created_at=now,
        updated_at=now
    )
    
    assert session.challenge_id == SAMPLE_CHALLENGE_ID
    assert session.challenge == SAMPLE_CHALLENGE_SECRET
    assert session.agent_did == SAMPLE_DID
    assert session.auth_token == "token"
    assert session.created_at == now
    assert session.updated_at == now

def test_client_agent_session_default_dates():
    """Test ClientAgentSession default dates"""
    session = ClientAgentSession(
        challenge_id=SAMPLE_CHALLENGE_ID,
        challenge=SAMPLE_CHALLENGE_SECRET
    )
    
    assert session.created_at is not None
    assert session.updated_at is not None
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)

def test_client_agent_session_updates_creation():
    """Test ClientAgentSessionUpdates creation"""
    updates = ClientAgentSessionUpdates(
        agent_did=SAMPLE_DID,
        auth_token="token"
    )
    
    assert updates.agent_did == SAMPLE_DID
    assert updates.auth_token == "token"

def test_client_agent_session_updates_partial():
    """Test ClientAgentSessionUpdates with partial updates"""
    updates = ClientAgentSessionUpdates(
        agent_did=SAMPLE_DID
    )
    
    assert updates.agent_did == SAMPLE_DID
    assert updates.auth_token is None 