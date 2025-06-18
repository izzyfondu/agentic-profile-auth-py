import base64
import json
from typing import TypeVar, Generic, Any

T = TypeVar('T')

def base64_to_base64url(b64: str) -> str:
    """Convert standard base64 to base64url format"""
    return b64.replace('+', '-').replace('/', '_').rstrip('=')

def base64url_to_base64(b64u: str) -> str:
    """Convert base64url to standard base64 format"""
    b64 = b64u.replace('-', '+').replace('_', '/')
    padding = len(b64) % 4
    if padding:
        b64 += '=' * (4 - padding)
    return b64

def base64url_to_bytes(b64u: str) -> bytes:
    """Convert base64url string to bytes"""
    return base64.b64decode(base64url_to_base64(b64u))

def bytes_to_base64url(data: bytes) -> str:
    """Convert bytes to base64url string"""
    return base64_to_base64url(base64.b64encode(data).decode('utf-8'))

def base64url_to_object(b64u: str) -> dict[str, Any]:
    """Convert base64url encoded JSON to Python object"""
    return json.loads(base64url_to_bytes(b64u).decode('utf-8'))

def object_to_base64url(obj: dict[str, Any]) -> str:
    """Convert Python object to base64url encoded JSON"""
    return bytes_to_base64url(json.dumps(obj).encode('utf-8')) 