cat > components/__init__.py << 'EOF'
"""
Components package for Audio Steganography App
"""

from .header import show_header, show_minimal_header, show_status_header
from .sidebar import show_sidebar
from .footer import show_footer

__all__ = [
    'show_header',
    'show_minimal_header',
    'show_status_header',
    'show_sidebar',
    'show_footer'
]

__version__ = "1.0.0"
EOF
