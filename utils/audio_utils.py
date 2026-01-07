"""
Audio Utility Functions
Audio processing, format conversion, and analysis tools for steganography
"""

import numpy as np
import wave
import io
import soundfile as sf
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """Audio processing utilities for steganography"""
    
    @staticmethod
    def load_audio(file_obj) -> Tuple[np.ndarray, int]:
        """
        Load audio file from various sources
        
        Args:
            file_obj: File object, bytes, or file path
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            if isinstance(file_obj, bytes):
                # Handle bytes input
                audio_bytes = io.BytesIO(file_obj)
                sample_rate, audio_data = wavfile.read(audio_bytes)
            elif hasattr(file_obj, 'read'):
                # Handle file-like object
                audio_bytes = io.BytesIO(file_obj.read())
                sample_rate, audio_data = wavfile.read(audio_bytes)
            else:
                # Assume it's a file path
                sample_rate, audio_data = wavfile.read(file_obj)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    @staticmethod
    def save_audio(audio_data: np.ndarray, sample_rate: int, 
                   output_path: str = None) -> bytes:
        """
        Save audio data to WAV format
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            output_path: Optional file path to save to
            
        Returns:
            Audio bytes if output_path is None
        """
        # Ensure proper data type
        if audio_data.dtype != np.int16:
            # Normalize to int16 range
            audio_data = AudioProcessor.normalize_to_int16(audio_data)
        
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_data)
        buffer.seek(0)
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
        
        return buffer.getvalue()
    
    @staticmethod
    def normalize_to_int16(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to int16 range (-32768 to 32767)"""
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            # Float audio in range [-1, 1]
            audio_data = np.clip(audio_data, -1, 1)
            audio_data = (audio_data * 32767).astype(np.int16)
        elif np.issubdtype(audio_data.dtype, np.integer):
            # Already integer, just convert to int16
            max_val = np.iinfo(audio_data.dtype).max
            min_val = np.iinfo(audio_data.dtype).min
            
            # Scale to int16 range
            if max_val > 32767 or min_val < -32768:
                scale_factor = 32767 / max(abs(max_val), abs(min_val))
                audio_data = (audio_data * scale_factor).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        return audio_data
    
    @staticmethod
    def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono by averaging channels"""
        if len(audio_data.shape) == 1:
            return audio_data  # Already mono
        
        if audio_data.shape[1] > 1:  # Multiple channels
            # Average channels
            return np.mean(audio_data, axis=1).astype(audio_data.dtype)
        
        return audio_data
    
    @staticmethod
    def get_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate audio duration in seconds"""
        return len(audio_data) / sample_rate
    
    @staticmethod
    def calculate_snr(original: np.ndarray, encoded: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio between original and encoded audio
        
        Args:
            original: Original audio samples
            encoded: Encoded audio samples
            
        Returns:
            SNR in decibels
        """
        # Ensure same length
        min_len = min(len(original), len(encoded))
        original = original[:min_len]
        encoded = encoded[:min_len]
        
        # Calculate signal and noise power
        signal_power = np.mean(original.astype(np.float64) ** 2)
        noise = encoded.astype(np.float64) - original.astype(np.float64)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, encoded: np.ndarray, 
                       max_value: int = 32767) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio
        
        Args:
            original: Original audio samples
            encoded: Encoded audio samples
            max_value: Maximum possible value (for 16-bit: 32767)
            
        Returns:
            PSNR in decibels
        """
        min_len = min(len(original), len(encoded))
        original = original[:min_len]
        encoded = encoded[:min_len]
        
        mse = np.mean((original.astype(np.float64) - encoded.astype(np.float64)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def analyze_lsb_pattern(audio_data: np.ndarray, bits: int = 1) -> Dict:
        """
        Analyze LSB patterns for steganalysis
        
        Args:
            audio_data: Audio samples to analyze
            bits: Number of LSB bits to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Use first channel
        
        results = {
            'lsb_distribution': {},
            'lsb_randomness': 0.0,
            'chi_square': 0.0,
            'rs_analysis': 0.0,
            'suspicious': False
        }
        
        # Extract LSBs
        mask = (1 << bits) - 1
        lsb_values = audio_data & mask
        
        # Analyze distribution
        unique, counts = np.unique(lsb_values, return_counts=True)
        for val, count in zip(unique, counts):
            results['lsb_distribution'][int(val)] = int(count)
        
        # Calculate randomness (entropy-like measure)
        total = len(lsb_values)
        if total > 0:
            probabilities = counts / total
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(2 ** bits)
            results['lsb_randomness'] = entropy / max_entropy
        
        # Chi-square test for uniform distribution
        if len(counts) > 1:
            expected = total / (2 ** bits)
            chi_square = np.sum((counts - expected) ** 2 / expected)
            results['chi_square'] = chi_square
            
            # High chi-square indicates non-uniform distribution (suspicious)
            results['suspicious'] = chi_square > (2 ** bits) * 10
        
        return results
    
    @staticmethod
    def detect_steganography(audio_data: np.ndarray, 
                            threshold: float = 0.3) -> Dict:
        """
        Detect potential steganography in audio
        
        Args:
            audio_data: Audio samples to analyze
            threshold: Suspicion threshold (0-1)
            
        Returns:
            Detection results
        """
        results = {
            'detected': False,
            'confidence': 0.0,
            'suspected_bits': 0,
            'analysis': {}
        }
        
        # Analyze different bit depths
        for bits in [1, 2, 3, 4]:
            analysis = AudioProcessor.analyze_lsb_pattern(audio_data, bits)
            results['analysis'][bits] = analysis
            
            # High randomness and uniform distribution suggests hidden data
            if analysis['lsb_randomness'] > 0.9 and analysis['chi_square'] < 10:
                confidence = analysis['lsb_randomness']
                if confidence > results['confidence']:
                    results['confidence'] = confidence
                    results['suspected_bits'] = bits
        
        results['detected'] = results['confidence'] > threshold
        
        return results
    
    @staticmethod
    def plot_audio_comparison(original: np.ndarray, encoded: np.ndarray,
                             sample_rate: int, title: str = "Audio Comparison") -> plt.Figure:
        """
        Create comparison plot of original vs encoded audio
        
        Args:
            original: Original audio samples
            encoded: Encoded audio samples
            sample_rate: Sample rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Time domain plot
        time = np.arange(len(original)) / sample_rate
        
        axes[0].plot(time[:1000], original[:1000], 'b-', alpha=0.7, label='Original')
        axes[0].plot(time[:1000], encoded[:1000], 'r-', alpha=0.7, label='Encoded')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Time Domain')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Difference plot
        diff = encoded.astype(np.float64) - original.astype(np.float64)
        axes[1].plot(time[:1000], diff[:1000], 'g-', alpha=0.7)
        axes[1].fill_between(time[:1000], diff[:1000], 0, alpha=0.3, color='green')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Difference')
        axes[1].set_title('Difference (Encoded - Original)')
        axes[1].grid(True, alpha=0.3)
        
        # LSB histogram
        original_lsb = original & 1
        encoded_lsb = encoded & 1
        
        axes[2].hist(original_lsb[:10000], bins=2, alpha=0.7, 
                    label='Original LSB', color='blue', rwidth=0.8)
        axes[2].hist(encoded_lsb[:10000], bins=2, alpha=0.7,
                    label='Encoded LSB', color='red', rwidth=0.8)
        axes[2].set_xlabel('LSB Value (0 or 1)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('LSB Distribution Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_frequency_analysis(original: np.ndarray, encoded: np.ndarray,
                               sample_rate: int) -> plt.Figure:
        """
        Create frequency domain analysis plot
        
        Args:
            original: Original audio samples
            encoded: Encoded audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate FFT
        n = min(len(original), 4096)
        freq = fftfreq(n, 1/sample_rate)[:n//2]
        
        original_fft = np.abs(fft(original[:n]))[:n//2]
        encoded_fft = np.abs(fft(encoded[:n]))[:n//2]
        diff_fft = np.abs(encoded_fft - original_fft)
        
        # Original spectrum
        axes[0, 0].semilogy(freq, original_fft, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].set_title('Original Audio Spectrum')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Encoded spectrum
        axes[0, 1].semilogy(freq, encoded_fft, 'r-', alpha=0.7)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Encoded Audio Spectrum')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Difference spectrum
        axes[1, 0].plot(freq, diff_fft, 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude Difference')
        axes[1, 0].set_title('Spectral Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        # SNR by frequency
        snr_freq = 10 * np.log10((original_fft ** 2) / ((diff_fft + 1e-10) ** 2))
        axes[1, 1].plot(freq, snr_freq, 'm-', alpha=0.7)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].set_title('Frequency-dependent SNR')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_analysis_report(original: np.ndarray, encoded: np.ndarray,
                                sample_rate: int, lsb_bits: int) -> Dict:
        """
        Generate comprehensive analysis report
        
        Args:
            original: Original audio samples
            encoded: Encoded audio samples
            sample_rate: Sample rate in Hz
            lsb_bits: Number of LSB bits used
            
        Returns:
            Dictionary with analysis report
        """
        report = {
            'basic_metrics': {},
            'quality_metrics': {},
            'steganalysis': {},
            'recommendations': []
        }
        
        # Basic metrics
        report['basic_metrics']['duration'] = len(original) / sample_rate
        report['basic_metrics']['samples'] = len(original)
        report['basic_metrics']['sample_rate'] = sample_rate
        report['basic_metrics']['lsb_bits_used'] = lsb_bits
        
        # Quality metrics
        report['quality_metrics']['snr'] = AudioProcessor.calculate_snr(original, encoded)
        report['quality_metrics']['psnr'] = AudioProcessor.calculate_psnr(original, encoded)
        
        mse = np.mean((original.astype(np.float64) - encoded.astype(np.float64)) ** 2)
        report['quality_metrics']['mse'] = mse
        report['quality_metrics']['rmse'] = np.sqrt(mse)
        
        # Steganalysis
        report['steganalysis'] = AudioProcessor.detect_steganography(encoded)
        
        # Recommendations
        if report['quality_metrics']['snr'] > 40:
            report['recommendations'].append("✅ Excellent quality - changes are inaudible")
        elif report['quality_metrics']['snr'] > 20:
            report['recommendations'].append("⚠️ Good quality - minimal audible impact")
        else:
            report['recommendations'].append("❌ Poor quality - audible artifacts likely")
        
        if report['steganalysis']['detected']:
            report['recommendations'].append("⚠️ Steganography may be detectable")
        else:
            report['recommendations'].append("✅ Steganography likely undetectable")
        
        if lsb_bits <= 2:
            report['recommendations'].append("✅ Low LSB bits - good for security")
        else:
            report['recommendations'].append("⚠️ High LSB bits - easier to detect")
        
        return report
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, original_rate: int, 
                       target_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio_data: Original audio samples
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if original_rate == target_rate:
            return audio_data
        
        # Calculate resampling ratio
        ratio = target_rate / original_rate
        
        # Use scipy's resample function
        num_samples = int(len(audio_data) * ratio)
        resampled = signal.resample(audio_data, num_samples)
        
        return resampled.astype(audio_data.dtype)
    
    @staticmethod
    def add_noise(audio_data: np.ndarray, noise_level: float = 0.001) -> np.ndarray:
        """
        Add Gaussian noise to audio
        
        Args:
            audio_data: Audio samples
            noise_level: Noise level (0-1)
            
        Returns:
            Noisy audio data
        """
        noise = np.random.normal(0, noise_level * 32767, len(audio_data))
        noisy_audio = audio_data.astype(np.float64) + noise
        return AudioProcessor.normalize_to_int16(noisy_audio)
    
    @staticmethod
    def trim_audio(audio_data: np.ndarray, sample_rate: int, 
                   start_time: float = 0, end_time: Optional[float] = None) -> np.ndarray:
        """
        Trim audio to specified time range
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of audio)
            
        Returns:
            Trimmed audio data
        """
        start_sample = int(start_time * sample_rate)
        if end_time is None:
            end_sample = len(audio_data)
        else:
            end_sample = int(end_time * sample_rate)
        
        return audio_data[start_sample:end_sample]
    
    @staticmethod
    def concatenate_audio(audio_list: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate multiple audio arrays
        
        Args:
            audio_list: List of audio arrays
            
        Returns:
            Concatenated audio
        """
        return np.concatenate(audio_list)
    
    @staticmethod
    def create_silence(duration: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Create silent audio of specified duration
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Silent audio samples
        """
        num_samples = int(duration * sample_rate)
        return np.zeros(num_samples, dtype=np.int16)
    
    @staticmethod
    def analyze_audio_quality(audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Analyze audio quality metrics
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {}
        
        # Convert to mono for analysis
        if len(audio_data.shape) > 1:
            audio_mono = AudioProcessor.convert_to_mono(audio_data)
        else:
            audio_mono = audio_data
        
        # Dynamic range
        max_amplitude = np.max(np.abs(audio_mono))
        metrics['peak_level'] = 20 * np.log10(max_amplitude / 32767) if max_amplitude > 0 else -np.inf
        
        # RMS level
        rms = np.sqrt(np.mean(audio_mono.astype(np.float64) ** 2))
        metrics['rms_level'] = 20 * np.log10(rms / 32767) if rms > 0 else -np.inf
        
        # Crest factor
        metrics['crest_factor'] = metrics['peak_level'] - metrics['rms_level']
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio_mono)))
        metrics['zero_crossing_rate'] = zero_crossings / len(audio_mono)
        
        # Spectral centroid
        if len(audio_mono) > 0:
            fft_vals = np.abs(fft(audio_mono[:4096]))
            freqs = fftfreq(4096, 1/sample_rate)
            positive_freqs = freqs[:2048]
            positive_fft = fft_vals[:2048]
            
            if np.sum(positive_fft) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                metrics['spectral_centroid'] = spectral_centroid
            else:
                metrics['spectral_centroid'] = 0
        
        return metrics

# Utility functions for Streamlit integration
def create_spectrogram_plot(audio_data: np.ndarray, sample_rate: int) -> plt.Figure:
    """
    Create spectrogram plot for audio
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Matplotlib figure with spectrogram
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = AudioProcessor.convert_to_mono(audio_data)
    
    # Create spectrogram
    NFFT = 1024
    Fs = sample_rate
    noverlap = NFFT // 2
    
    Pxx, freqs, bins, im = ax.specgram(audio_data, NFFT=NFFT, Fs=Fs, 
                                       noverlap=noverlap, cmap='viridis')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Audio Spectrogram')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Intensity (dB)')
    
    return fig

def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string"""
    return base64.b64encode(audio_bytes).decode('utf-8')

def base64_to_audio(base64_string: str) -> bytes:
    """Convert base64 string to audio bytes"""
    return base64.b64decode(base64_string)

def validate_audio_file(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate audio file
    
    Args:
        file_bytes: Audio file bytes
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Try to load the audio
        audio_data, sample_rate = AudioProcessor.load_audio(file_bytes)
        
        # Check if it's WAV format
        if not isinstance(audio_data, np.ndarray):
            return False, "Invalid audio format"
        
        # Check duration
        duration = len(audio_data) / sample_rate
        if duration < 0.1:  # Less than 100ms
            return False, "Audio too short (minimum 0.1 seconds)"
        
        if duration > 300:  # More than 5 minutes
            return False, "Audio too long (maximum 5 minutes)"
        
        # Check sample rate
        if sample_rate < 8000 or sample_rate > 192000:
            return False, f"Unsupported sample rate: {sample_rate} Hz"
        
        return True, f"Valid audio: {duration:.2f}s, {sample_rate}Hz"
    
    except Exception as e:
        return False, f"Invalid audio file: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Test the audio processor
    print("Audio Processor Test")
    
    # Create test audio
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = (32767 * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    
    # Test functions
    print(f"Duration: {AudioProcessor.get_audio_duration(test_audio, sample_rate)}s")
    
    # Test SNR calculation (identical audio should have infinite SNR)
    snr = AudioProcessor.calculate_snr(test_audio, test_audio)
    print(f"SNR (identical): {snr} dB")
    
    # Test with noise
    noisy_audio = AudioProcessor.add_noise(test_audio, 0.01)
    snr = AudioProcessor.calculate_snr(test_audio, noisy_audio)
    print(f"SNR (with noise): {snr:.2f} dB")
    
    # Test analysis
    analysis = AudioProcessor.analyze_lsb_pattern(test_audio)
    print(f"LSB Randomness: {analysis['lsb_randomness']:.3f}")
