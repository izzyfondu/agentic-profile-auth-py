from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Type aliases
DID = str  # May include a fragment... or not
FragmentID = str  # May be full DID, or just the fragment part, such as "#key-7"
UserID = Union[str, int]

class VerificationMethod(BaseModel):
    """Verification method from DID document"""
    id: str
    type: str
    controller: str
    public_key_jwk: Optional[Dict[str, Any]] = None

class Service(BaseModel):
    """Base service from DID document"""
    id: str
    type: str
    service_endpoint: str

class AgentService(Service):
    """Agent service with capability invocation"""
    name: str
    capability_invocation: List[Union[FragmentID, VerificationMethod]]

class AgenticProfile(BaseModel):
    """Agentic Profile extending DID document"""
    id: DID
    name: str
    verification_method: Optional[List[VerificationMethod]] = None
    service: Optional[List[AgentService]] = None
    ttl: Optional[int] = Field(default=86400)  # TTL in seconds, default is one day

class AgenticChallenge(BaseModel):
    """Challenge for authentication"""
    type: str = "agentic-challenge"
    challenge: Dict[str, str]

class AgenticJwsHeader(BaseModel):
    """JWS header for agentic authentication"""
    alg: str = "EdDSA"
    typ: str = "JWT"

class AgenticJwsPayload(BaseModel):
    """JWS payload for agentic authentication"""
    challenge: Dict[str, str]
    attest: Dict[str, str]

class ClientAgentSession(BaseModel):
    """Client agent session for authentication"""
    challenge_id: str
    challenge: str
    agent_did: Optional[DID] = None
    auth_token: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ClientAgentSessionUpdates(BaseModel):
    """Updates for client agent session"""
    agent_did: Optional[DID] = None
    auth_token: Optional[str] = None 