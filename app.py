# Save this as app.py - ONLY Python code, no bash commands!
"""
Audio Steganography Tool
Clean working version
"""

import streamlit as st
import numpy as np
from scipy.io import wavfile
import io
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Audio Steganography",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add some CSS styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stFileUploader > div {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔊 Audio Steganography Tool")
st.markdown("Hide and extract secret messages in audio files using LSB (Least Significant Bit) technique")

# Simple LSB encoding function
def encode_audio_lsb(audio_data, message, bits=1):
    """Encode a message into audio using LSB"""
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_flat = audio_data[:, 0].copy()
        was_stereo = True
    else:
        audio_flat = audio_data.copy()
        was_stereo = False
    
    # Add end marker
    message += "###END###"
    
    # Convert message to binary
    binary_message = ''
    for char in message:
        binary_message += format(ord(char), '08b')
    
    # Check if audio is long enough
    if len(binary_message) > len(audio_flat) * bits:
        raise ValueError(f"Message too long! Audio can hold {len(audio_flat) * bits // 8} characters max")
    
    # Encode the message
    encoded_audio = audio_flat.copy()
    mask = ~((1 << bits) - 1)  # Mask to clear LSB bits
    
    for i in range(len(encoded_audio)):
        if i * bits >= len(binary_message):
            break
        
        # Clear LSB bits
        encoded_audio[i] = encoded_audio[i] & mask
        
        # Set new LSB bits
        for b in range(bits):
            if i * bits + b < len(binary_message):
                bit_value = int(binary_message[i * bits + b])
                encoded_audio[i] |= (bit_value << b)
    
    # Return in original format
    if was_stereo:
        audio_data[:, 0] = encoded_audio
        return audio_data
    else:
        return encoded_audio

# Simple LSB decoding function
def decode_audio_lsb(audio_data, bits=1):
    """Decode a message from audio using LSB"""
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_flat = audio_data[:, 0]
    else:
        audio_flat = audio_data
    
    # Extract LSB bits
    binary_message = ''
    for sample in audio_flat:
        for b in range(bits):
            bit = (sample >> b) & 1
            binary_message += str(bit)
    
    # Convert binary to text
    message = ''
    for i in range(0, len(binary_message), 8):
        if i + 8 <= len(binary_message):
            byte = binary_message[i:i+8]
            try:
                char = chr(int(byte, 2))
                message += char
                
                # Check for end marker
                if message.endswith("###END###"):
                    return message[:-8]  # Remove end marker
            except:
                continue
    
    return None

# Calculate capacity
def calculate_capacity(audio_data, bits):
    """Calculate maximum message capacity"""
    if len(audio_data.shape) > 1:
        samples = audio_data.shape[0]
    else:
        samples = len(audio_data)
    
    # Account for end marker (8 characters = 64 bits)
    return max(0, (samples * bits) // 8 - 8)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # LSB bits selection
    lsb_bits = st.slider(
        "LSB Bits",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of LSB bits to use per sample (1-4)"
    )
    
    # Quality indicator
    if lsb_bits == 1:
        st.success("✅ Excellent quality - undetectable")
    elif lsb_bits == 2:
        st.warning("⚠️ Good quality - minimal impact")
    elif lsb_bits == 3:
        st.info("ℹ️ Fair quality - some impact")
    else:
        st.error("🔴 Noticeable impact - use with caution")
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload an audio file (WAV format)
    2. Enter your secret message
    3. Tool modifies least significant bits
    4. Download the encoded audio
    5. Upload encoded audio to extract message
    """)

# Create main tabs
tab1, tab2 = st.tabs(["🔒 Encode Message", "🔓 Decode Message"])

# Encode Tab
with tab1:
    st.header("Hide Your Secret Message")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Audio File")
        audio_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav'],
            help="Select a WAV audio file to hide your message in",
            key="encode_uploader"
        )
        
        if audio_file:
            # Read the audio file
            audio_bytes = audio_file.read()
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            
            # Display audio player
            st.audio(audio_bytes, format='audio/wav')
            
            # Calculate and show audio info
            duration = len(audio_data) / sample_rate
            capacity = calculate_capacity(audio_data, lsb_bits)
            
            with st.expander("📊 Audio Information"):
                st.write(f"**File name:** {audio_file.name}")
                st.write(f"**Duration:** {duration:.2f} seconds")
                st.write(f"**Sample rate:** {sample_rate} Hz")
                st.write(f"**Samples:** {len(audio_data):,}")
                st.write(f"**Max message capacity:** ~{capacity} characters")
            
            # Store for encoding
            st.session_state.encode_audio_data = audio_data
            st.session_state.encode_sample_rate = sample_rate
    
    with col2:
        st.subheader("2. Enter Secret Message")
        
        secret_message = st.text_area(
            "Your secret message:",
            height=150,
            placeholder="Type your confidential message here...",
            help="This message will be hidden in the audio file"
        )
        
        # Show character count if message is entered
        if secret_message:
            if 'encode_audio_data' in st.session_state:
                capacity = calculate_capacity(st.session_state.encode_audio_data, lsb_bits)
                st.caption(f"Characters: {len(secret_message)} / {capacity}")
                
                if len(secret_message) > capacity:
                    st.error(f"⚠️ Message too long! Maximum capacity is {capacity} characters")
        
        # Encode button
        encode_button = st.button(
            "🚀 Encode Message into Audio",
            type="primary",
            use_container_width=True,
            disabled=not (audio_file and secret_message)
        )
        
        if encode_button and audio_file and secret_message:
            try:
                with st.spinner("Encoding your secret message..."):
                    # Encode the message
                    encoded_audio = encode_audio_lsb(
                        st.session_state.encode_audio_data,
                        secret_message,
                        lsb_bits
                    )
                    
                    # Create download button
                    buffer = BytesIO()
                    wavfile.write(buffer, st.session_state.encode_sample_rate, 
                                encoded_audio.astype(np.int16))
                    buffer.seek(0)
                    
                    st.success("✅ Message successfully encoded!")
                    
                    # Download section
                    st.subheader("3. Download Encoded Audio")
                    
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.download_button(
                            label="💾 Download Encoded Audio",
                            data=buffer.getvalue(),
                            file_name="secret_audio.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                    
                    with col_b:
                        st.audio(buffer, format='audio/wav')
                    
                    # Show encoding details
                    with st.expander("📝 Encoding Details"):
                        st.write(f"**Message length:** {len(secret_message)} characters")
                        st.write(f"**LSB bits used:** {lsb_bits}")
                        st.write(f"**Audio duration:** {len(st.session_state.encode_audio_data)/st.session_state.encode_sample_rate:.2f}s")
                        st.write(f"**Quality impact:** {'Minimal' if lsb_bits <= 2 else 'Noticeable'}")
            
            except ValueError as e:
                st.error(f"❌ Encoding error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Decode Tab
with tab2:
    st.header("Extract Hidden Message")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Encoded Audio")
        encoded_file = st.file_uploader(
            "Choose encoded WAV file",
            type=['wav'],
            help="Upload an audio file containing a hidden message",
            key="decode_uploader"
        )
        
        if encoded_file:
            # Read and display audio
            audio_bytes = encoded_file.read()
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            
            st.audio(audio_bytes, format='audio/wav')
    
    with col2:
        st.subheader("2. Extract Settings")
        
        # Bits selection for decoding (can be different)
        decode_bits = st.slider(
            "LSB Bits for decoding:",
            min_value=1,
            max_value=4,
            value=lsb_bits,
            key="decode_bits"
        )
        
        # Decode button
        decode_button = st.button(
            "🔍 Extract Hidden Message",
            type="primary",
            use_container_width=True,
            disabled=not encoded_file
        )
        
        if decode_button and encoded_file:
            try:
                with st.spinner("Extracting hidden message..."):
                    # Decode the message
                    decoded_message = decode_audio_lsb(audio_data, decode_bits)
                    
                    # Display results
                    if decoded_message:
                        st.success("✅ Hidden message found!")
                        
                        # Display the message
                        st.subheader("📜 Extracted Message:")
                        st.text_area(
                            "Secret Message",
                            decoded_message,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        
                        # Show decoding details
                        with st.expander("🔍 Decoding Details"):
                            st.write(f"**Message length:** {len(decoded_message)} characters")
                            st.write(f"**LSB bits used:** {decode_bits}")
                            st.write(f"**Audio duration:** {len(audio_data)/sample_rate:.2f}s")
                            st.write("**Status:** Successfully extracted")
                    
                    else:
                        st.warning("⚠️ No hidden message found!")
                        
                        st.info("""
                        **Possible reasons:**
                        1. Wrong LSB bits setting
                        2. File doesn't contain a hidden message
                        3. Audio file was modified
                        4. Different encoding method was used
                        
                        **Try:**
                        - Adjust the LSB bits slider
                        - Ensure you're using the correct encoded file
                        - Try different bit values (1-4)
                        """)
            
            except Exception as e:
                st.error(f"❌ Error during decoding: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p>🔊 Audio Steganography Tool v1.0 | Made with ❤️ using Streamlit</p>
        <p style="font-size: 0.9rem;">
            For educational purposes only. Always respect privacy and copyright laws.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Run the app
if __name__ == "__main__":
    pass
