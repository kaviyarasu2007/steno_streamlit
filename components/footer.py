"""
Footer Component for Audio Steganography App
"""

import streamlit as st
import datetime

def show_footer():
    """
    Display the application footer with copyright and links
    """
    
    # Get current year
    current_year = datetime.datetime.now().year
    
    footer_html = f"""
    <div style="
        text-align: center;
        padding: 2rem 1rem;
        color: #94a3b8;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    ">
        <p>&copy; {current_year} Audio Steganography Pro</p>
        <p>Version 2.1.0 • For educational purposes only</p>
        <div style="margin-top: 1rem; opacity: 0.7;">
            🔒 Privacy • ⚖️ Terms • 📧 Contact
        </div>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)
