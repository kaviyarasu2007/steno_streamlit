cat > components/sidebar.py << 'EOF'
"""
Sidebar Component for Audio Steganography App
"""

import streamlit as st

def show_sidebar():
    """
    Display the sidebar with settings and controls
    
    Returns:
        Dictionary of current settings
    """
    
    with st.sidebar:
        # Sidebar header with logo
        st.markdown("""
        <div style="
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
        ">
            <div style="
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            ">
                ⚙️
            </div>
            <h2 style="
                margin: 0;
                color: #e2e8f0;
                font-size: 1.5rem;
            ">
                Control Panel
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Steganography Settings Section
        st.markdown("### 🛠️ Steganography Settings")
        
        # LSB Bits Selection
        lsb_bits = st.slider(
            "LSB Bits",
            min_value=1,
            max_value=8,
            value=2,
            help="Number of least significant bits to modify per sample"
        )
        
        # Quality indicator
        if lsb_bits == 1:
            st.success("✅ Excellent Quality - Undetectable")
        elif lsb_bits == 2:
            st.warning("⚠️ Good Quality - Minimal Impact")
        elif lsb_bits <= 4:
            st.info("ℹ️ Fair Quality - Some Impact")
        else:
            st.error("🔴 Poor Quality - Noticeable Impact")
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("⚡ Advanced Settings"):
            sampling_method = st.selectbox(
                "Sampling Method",
                ["Sequential", "Random", "Interleaved"],
                help="Method for distributing bits in audio"
            )
            
            error_correction = st.checkbox(
                "Enable Error Correction",
                value=True,
                help="Add redundancy for better recovery"
            )
            
            add_noise = st.checkbox(
                "Add Random Noise",
                value=False,
                help="Add noise to reduce detectability"
            )
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📚 Guide", use_container_width=True):
                st.session_state['active_tab'] = "Guide"
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        # System Info
        st.markdown("### 📊 System Information")
        
        st.caption(f"LSB Bits: {lsb_bits}")
        st.caption(f"Sampling: {sampling_method}")
        st.caption("Status: ✅ Ready")
        
        # Version info
        st.markdown("---")
        st.caption("Version 2.1.0 | © 2024")
    
    # Return settings as dictionary
    return {
        'lsb_bits': lsb_bits,
        'sampling_method': sampling_method,
        'error_correction': error_correction,
        'add_noise': add_noise
    }
EOF
