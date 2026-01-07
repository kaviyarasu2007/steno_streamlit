# 🔊 Audio Steganography Pro

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

A sophisticated web application for hiding and extracting secret messages in audio files using LSB (Least Significant Bit) steganography techniques.

![App Screenshot](assets/images/screenshot.png)

## ✨ Features

### 🔒 **Encoding**
- Hide text messages in WAV audio files
- Adjustable LSB bits (1-8 bits per sample)
- AES-256 encryption for added security
- Message compression to maximize capacity
- Error correction for reliable extraction
- Real-time quality analysis

### 🔓 **Decoding**
- Extract hidden messages from encoded audio
- Automatic bit-depth detection
- Password-protected extraction
- Validation and error checking
- Steganalysis detection

### 📊 **Analysis Tools**
- Signal-to-Noise Ratio (SNR) calculation
- Peak Signal-to-Noise Ratio (PSNR)
- LSB pattern analysis
- Audio quality metrics
- Visual comparison tools

### 🎨 **User Experience**
- Modern, responsive Streamlit interface
- Real-time progress indicators
- Audio preview and playback
- Detailed statistics and reporting
- Comprehensive user guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/audio-steganography.git
   cd audio-steganography
