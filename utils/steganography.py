cat > utils/steganography.py << 'EOF'
"""
Simple Audio Steganography Engine
Working version for basic LSB steganography
"""

import numpy as np
import hashlib
import base64
from cryptography.fernet import Fernet

class AudioSteganography:
    """Simple Audio Steganography Engine"""
    
    def __init__(self):
        self.version = "1.0"
    
    def calculate_capacity(self, audio_data: np.ndarray, bits: int) -> int:
        """Calculate maximum message capacity in characters"""
        if len(audio_data.shape) > 1:
            samples = audio_data.shape[0]
        else:
            samples = len(audio_data)
        
        # Account for end marker (8 chars = 64 bits)
        available_bits = samples * bits
        return max(0, available_bits // 8 - 8)
    
    def encode(self, audio_data: np.ndarray, message: str, bits: int = 2,
               password: str = None, compress: bool = False) -> np.ndarray:
        """
        Encode a message into audio data
        
        Args:
            audio_data: Original audio samples
            message: Secret message to hide
            bits: Number of LSB bits to use (1-4)
            password: Optional encryption password
            compress: Whether to compress the message
            
        Returns:
            Encoded audio data
        """
        # Validate inputs
        if bits < 1 or bits > 4:
            raise ValueError("Bits must be between 1 and 4")
        
        if len(message) == 0:
            raise ValueError("Message cannot be empty")
        
        # Process audio (use first channel if stereo)
        if len(audio_data.shape) > 1:
            audio_flat = audio_data[:, 0].copy()
            stereo = True
        else:
            audio_flat = audio_data.copy()
            stereo = False
        
        # Add end marker
        processed_message = message + "###END###"
        
        # Optional encryption
        if password:
            # Simple XOR encryption for demo (in real app use proper encryption)
            key = hashlib.sha256(password.encode()).digest()
            encrypted_bytes = []
            for i, char in enumerate(processed_message):
                key_byte = key[i % len(key)]
                encrypted_bytes.append(ord(char) ^ key_byte)
            processed_message = base64.b64encode(bytes(encrypted_bytes)).decode('utf-8')
        
        # Convert message to binary
        binary_message = ''.join(format(ord(c), '08b') for c in processed_message)
        
        # Check capacity
        if len(binary_message) > len(audio_flat) * bits:
            max_chars = self.calculate_capacity(audio_data, bits)
            raise ValueError(f"Message too long. Max: {max_chars} chars, Got: {len(message)}")
        
        # Encode message into LSBs
        encoded_audio = self._embed_bits(audio_flat, binary_message, bits)
        
        # Restore stereo if needed
        if stereo:
            audio_data[:, 0] = encoded_audio
            return audio_data
        else:
            return encoded_audio
    
    def decode(self, audio_data: np.ndarray, bits: int = 2,
               password: str = None) -> str:
        """
        Decode a message from audio data
        
        Args:
            audio_data: Audio samples potentially containing hidden message
            bits: Number of LSB bits used during encoding
            password: Optional decryption password
            
        Returns:
            Decoded message or empty string if no message found
        """
        # Process audio (use first channel if stereo)
        if len(audio_data.shape) > 1:
            audio_flat = audio_data[:, 0]
        else:
            audio_flat = audio_data
        
        # Extract bits
        binary_data = self._extract_bits(audio_flat, bits)
        
        # Convert to characters
        chars = []
        for i in range(0, len(binary_data), 8):
            if i + 8 <= len(binary_data):
                byte = binary_data[i:i+8]
                try:
                    char = chr(int(byte, 2))
                    chars.append(char)
                    
                    # Check for end marker
                    if ''.join(chars[-8:]) == "###END###":
                        message = ''.join(chars[:-8])
                        
                        # Optional decryption
                        if password:
                            try:
                                decoded_bytes = base64.b64decode(message)
                                key = hashlib.sha256(password.encode()).digest()
                                decrypted_chars = []
                                for i, byte in enumerate(decoded_bytes):
                                    key_byte = key[i % len(key)]
                                    decrypted_chars.append(chr(byte ^ key_byte))
                                message = ''.join(decrypted_chars)
                            except:
                                return "Decryption failed - wrong password?"
                        
                        return message
                except:
                    continue
        
        return "No message found"
    
    def _embed_bits(self, audio: np.ndarray, binary_data: str, bits: int) -> np.ndarray:
        """Embed binary data into audio LSBs"""
        encoded_audio = audio.copy()
        mask = ~((1 << bits) - 1)
        
        for i in range(len(encoded_audio)):
            if i * bits >= len(binary_data):
                break
            
            # Clear LSB bits
            encoded_audio[i] = encoded_audio[i] & mask
            
            # Set new bits
            for b in range(bits):
                if i * bits + b < len(binary_data):
                    bit_value = int(binary_data[i * bits + b])
                    encoded_audio[i] |= (bit_value << b)
        
        return encoded_audio
    
    def _extract_bits(self, audio: np.ndarray, bits: int) -> str:
        """Extract binary data from audio LSBs"""
        binary_data = []
        
        for sample in audio:
            for b in range(bits):
                bit = (sample >> b) & 1
                binary_data.append(str(bit))
        
        return ''.join(binary_data)
EOF
