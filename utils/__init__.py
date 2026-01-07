cat > utils/__init__.py << 'EOF'
"""
Utilities package for Audio Steganography App
"""

from .steganography import AudioSteganography

__all__ = [
    'AudioSteganography'
]

__version__ = "1.0.0"
EOF
