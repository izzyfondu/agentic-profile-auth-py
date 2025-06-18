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
    controller: Optional[str] = None  # Made optional to handle real DID documents
    public_key_jwk: Optional[Dict[str, Any]] = Field(default=None, alias="publicKeyJwk")
    
    class Config:
        populate_by_name = True

class Service(BaseModel):
    """Base service from DID document"""
    id: str
    type: str
    service_endpoint: str = Field(alias="serviceEndpoint")
    
    class Config:
        populate_by_name = True

class AgentService(Service):
    """Agent service with capability invocation"""
    name: str
    capability_invocation: List[Union[FragmentID, VerificationMethod]] = Field(alias="capabilityInvocation")
    
    class Config:
        populate_by_name = True

class AgenticProfile(BaseModel):
    """Agentic Profile extending DID document"""
    id: DID
    name: Optional[str] = None  # Made optional to handle real DID documents
    verification_method: Optional[List[VerificationMethod]] = Field(default=None, alias="verificationMethod")
    service: Optional[List[AgentService]] = None
    ttl: Optional[int] = Field(default=86400)  # TTL in seconds, default is one day
    
    class Config:
        populate_by_name = True

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
    challenge: Union[str, Dict[str, str]]  # Can be string or object
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

class InMemoryAgenticProfileStore:
    """
    Simple in-memory store for caching agentic profiles.
    
    This is suitable for testing, examples, and local development.
    Not suitable for production use as data is lost when the process exits.
    """
    
    def __init__(self):
        self._profiles: Dict[str, AgenticProfile] = {}
    
    async def load_agentic_profile(self, did: str) -> Optional[AgenticProfile]:
        """
        Load an agentic profile from the store
        Args:
            did: The DID to load
        Returns:
            Optional[AgenticProfile]: The profile if found, None otherwise
        """
        return self._profiles.get(did)
    
    async def save_agentic_profile(self, profile: AgenticProfile) -> None:
        """
        Save an agentic profile to the store
        Args:
            profile: The profile to save
        """
        self._profiles[profile.id] = profile
    
    def clear(self) -> None:
        """Clear all stored profiles"""
        self._profiles.clear()
    
    def __len__(self) -> int:
        """Return the number of stored profiles"""
        return len(self._profiles) 