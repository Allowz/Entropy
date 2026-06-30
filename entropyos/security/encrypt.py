from __future__ import annotations

import base64
import hashlib
from typing import Any

from entropyos.models import Memory


class MemoryEncryptor:
    def __init__(self, key: str = ""):
        self.key = key
        self._cipher: Any = None

    def _lazy_init(self) -> None:
        if self._cipher is None and self.key:
            try:
                from cryptography.fernet import Fernet
                derived = base64.urlsafe_b64encode(
                    hashlib.sha256(self.key.encode()).digest()
                )
                self._cipher = Fernet(derived)
            except ImportError:
                self._cipher = None

    def encrypt(self, memory: Memory) -> Memory:
        if not self.key or not memory.content:
            return memory
        self._lazy_init()
        if self._cipher is None:
            return memory
        encrypted = self._cipher.encrypt(memory.content.encode())
        memory.content = encrypted.decode()
        memory.metadata["encrypted"] = True
        return memory

    def decrypt(self, memory: Memory) -> Memory:
        if not self.key or not memory.content:
            return memory
        if not memory.metadata.get("encrypted"):
            return memory
        self._lazy_init()
        if self._cipher is None:
            return memory
        decrypted = self._cipher.decrypt(memory.content.encode())
        memory.content = decrypted.decode()
        memory.metadata["encrypted"] = False
        return memory

    def encrypt_str(self, text: str) -> str:
        if not self.key or not text:
            return text
        self._lazy_init()
        if self._cipher is None:
            return text
        return self._cipher.encrypt(text.encode()).decode()

    def decrypt_str(self, text: str) -> str:
        if not self.key or not text:
            return text
        self._lazy_init()
        if self._cipher is None:
            return text
        return self._cipher.decrypt(text.encode()).decode()
