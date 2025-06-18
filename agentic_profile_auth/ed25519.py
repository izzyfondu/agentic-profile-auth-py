from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from .b64u import base64url_to_bytes, bytes_to_base64url

async def verify(signature: str, message: str, public_key: str) -> bool:
    """
    Verify an Ed25519 signature
    
    Args:
        signature: Base64url encoded signature
        message: Message to verify
        public_key: Base64url encoded public key
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Convert inputs to bytes
        signature_bytes = base64url_to_bytes(signature)
        message_bytes = message.encode('utf-8')
        public_key_bytes = base64url_to_bytes(public_key)
        
        # Create Ed25519 public key
        public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        
        # Verify signature
        public_key_obj.verify(signature_bytes, message_bytes)
        return True
    except Exception:
        return False

def create_key_pair() -> tuple[str, str]:
    """
    Create a new Ed25519 key pair
    
    Returns:
        tuple[str, str]: (public_key, private_key) as base64url encoded strings
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Get raw bytes
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    # Convert to base64url
    private_b64u = bytes_to_base64url(private_bytes)
    public_b64u = bytes_to_base64url(public_bytes)
    
    return public_b64u, private_b64u 