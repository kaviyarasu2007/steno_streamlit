cat > components/header.py << 'EOF'
"""
Header Component for Audio Steganography App
"""

import streamlit as st

def show_header():
    """
    Display the application header with title and description
    """
    
    # Main header with styling
    header_html = """
    <div style="
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        color: white;
    ">
        <h1 style="
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            font-weight: 800;
            background: linear-gradient(45deg, #ffffff, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            🔊 Audio Steganography Pro
        </h1>
        
        <p style="
            font-size: 1.3rem;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto 1.5rem;
            line-height: 1.6;
        ">
            Hide secret messages in audio files using advanced LSB steganography techniques.
            Secure, undetectable, and easy to use.
        </p>
        
        <div style="
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        ">
            <span style="
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                padding: 0.7rem 1.5rem;
                border-radius: 50px;
                font-size: 1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                border: 2px solid rgba(255, 255, 255, 0.3);
            ">
                🔒 Military-grade Security
            </span>
            
            <span style="
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                padding: 0.7rem 1.5rem;
                border-radius: 50px;
                font-size: 1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                border: 2px solid rgba(255, 255, 255, 0.3);
            ">
                🎵 Lossless Audio Quality
            </span>
            
            <span style="
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                padding: 0.7rem 1.5rem;
                border-radius: 50px;
                font-size: 1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                border: 2px solid rgba(255, 255, 255, 0.3);
            ">
                ⚡ Real-time Processing
            </span>
        </div>
        
        <div style="
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.9rem;
            opacity: 0.8;
        ">
            <strong>Version 2.1.0</strong> • Supports WAV format • 1-8 LSB bits • AES-256 Encryption
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Quick navigation bar
    nav_html = """
    <div style="
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    ">
        <a href="#encode" style="
            text-decoration: none;
            color: #667eea;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: #f0f4ff;
            font-weight: 600;
            transition: all 0.3s;
        " onmouseover="this.style.background='#e0e7ff'" onmouseout="this.style.background='#f0f4ff'">
            🔒 Encode Message
        </a>
        
        <a href="#decode" style="
            text-decoration: none;
            color: #764ba2;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: #f9f0ff;
            font-weight: 600;
            transition: all 0.3s;
        " onmouseover="this.style.background='#f3e8ff'" onmouseout="this.style.background='#f9f0ff'">
            🔓 Decode Message
        </a>
        
        <a href="#guide" style="
            text-decoration: none;
            color: #10b981;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: #f0fdf4;
            font-weight: 600;
            transition: all 0.3s;
        " onmouseover="this.style.background='#dcfce7'" onmouseout="this.style.background='#f0fdf4'">
            📚 User Guide
        </a>
    </div>
    """
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")

# Alternative minimal header (optional)
def show_minimal_header():
    """
    Display a minimal header version
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1>🔊 Audio Steganography</h1>
            <p>Hide messages in audio files</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

# Status header for showing current operation
def show_status_header(status: str, icon: str = "ℹ️"):
    """
    Display a status header for current operation
    
    Args:
        status: Status message
        icon: Status icon
    """
    status_html = f"""
    <div style="
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    ">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 2rem;">{icon}</span>
            <div>
                <h4 style="margin: 0; color: #1e293b;">Current Status</h4>
                <p style="margin: 0.5rem 0 0 0; color: #475569;">{status}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(status_html, unsafe_allow_html=True)
EOF
