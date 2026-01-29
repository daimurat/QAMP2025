"""
API Key Encryption Utilities
"""
import base64
from cryptography.fernet import Fernet


def save_encrypted_key(encrypted_key: str, username: str, suffix: str = "_encrypted_api_key") -> bool:
    """Save encrypted key to file
    
    Args:
        encrypted_key: Encrypted API key
        username: Username (used as filename prefix)
        suffix: Additional suffix for file naming (e.g., provider identifier)
        
    Returns:
        True if save successful, False otherwise
    """
    if not username:
        username = 'anon'
    try:
        filename = f"{username}{suffix}"
        with open(filename, "w") as f:
            f.write(encrypted_key)
        return True
    except Exception:
        return False


def load_encrypted_key(username: str, suffix: str = "_encrypted_api_key") -> str | None:
    """Load encrypted key from file
    
    Args:
        username: Username (used as filename prefix)
        suffix: Additional suffix for file naming (e.g., provider identifier)
        
    Returns:
        Encrypted API key, or None if file doesn't exist
    """
    if not username:
        username = 'anon'
    try:
        filename = f"{username}{suffix}"
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def create_fernet(password: str) -> Fernet:
    """Create Fernet instance from password
    
    Args:
        password: Password used for encryption/decryption
        
    Returns:
        Fernet instance
    """
    key = base64.urlsafe_b64encode(password.ljust(32)[:32].encode())
    return Fernet(key)

# Made with Bob
