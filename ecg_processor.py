"""
ECG Signal Processor
Handles signal preprocessing and PQRST peak detection using NeuroKit2
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    import neurokit2 as nk
except ImportError:
    nk = None

from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d


class ECGProcessor:
    """Process ECG signals and detect PQRST peaks"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize ECG Processor
        
        Args:
            sampling_rate: Sampling frequency in Hz (default 250)
        """
        self.sampling_rate = sampling_rate
    
    def preprocess(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal: bandpass filter and baseline correction
        
        Args:
            ecg_signal: Raw ECG signal array
            
        Returns:
            Cleaned ECG signal
        """
        # Convert to numpy array if needed
        ecg_signal = np.array(ecg_signal, dtype=np.float64)
        
        # Handle edge cases
        if len(ecg_signal) < 10:
            return ecg_signal
        
        # Remove DC offset
        ecg_signal = ecg_signal - np.mean(ecg_signal)
        
        # Bandpass filter (0.5 - 40 Hz)
        try:
            ecg_signal = self._bandpass_filter(ecg_signal, 0.5, 40)
        except Exception:
            pass
        
        # Baseline wander removal using moving average
        try:
            baseline = uniform_filter1d(ecg_signal, size=int(0.2 * self.sampling_rate))
            ecg_signal = ecg_signal - baseline
        except Exception:
            pass
        
        # Normalize
        max_val = np.max(np.abs(ecg_signal))
        if max_val > 0:
            ecg_signal = ecg_signal / max_val
        
        return ecg_signal
    
    def _bandpass_filter(self, signal: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure valid filter parameters
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def detect_peaks(self, ecg_signal: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect PQRST peaks in ECG signal
        
        Args:
            ecg_signal: Preprocessed ECG signal
            
        Returns:
            Dictionary with peak indices for P, Q, R, S, T waves
        """
        ecg_clean = self.preprocess(ecg_signal)
        
        # Try NeuroKit2 first (more accurate)
        if nk is not None:
            try:
                return self._detect_peaks_neurokit(ecg_clean)
            except Exception:
                pass
        
        # Fallback to custom detection
        return self._detect_peaks_custom(ecg_clean)
    
    def _detect_peaks_neurokit(self, ecg_clean: np.ndarray) -> Dict[str, List[int]]:
        """Detect peaks using NeuroKit2"""
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=self.sampling_rate)
        r_peaks = rpeaks.get('ECG_R_Peaks', [])
        
        if len(r_peaks) == 0:
            return self._empty_peaks()
        
        # Delineate all waves
        try:
            _, waves = nk.ecg_delineate(
                ecg_clean, 
                rpeaks, 
                sampling_rate=self.sampling_rate,
                method="dwt"
            )
            
            p_peaks = self._clean_peaks(waves.get('ECG_P_Peaks', []))
            q_peaks = self._clean_peaks(waves.get('ECG_Q_Peaks', []))
            s_peaks = self._clean_peaks(waves.get('ECG_S_Peaks', []))
            t_peaks = self._clean_peaks(waves.get('ECG_T_Peaks', []))
            
        except Exception:
            # If delineation fails, estimate other peaks from R-peaks
            p_peaks, q_peaks, s_peaks, t_peaks = self._estimate_waves_from_r(r_peaks, len(ecg_clean))
        
        return {
            "p_peaks": list(map(int, p_peaks)),
            "q_peaks": list(map(int, q_peaks)),
            "r_peaks": list(map(int, r_peaks)),
            "s_peaks": list(map(int, s_peaks)),
            "t_peaks": list(map(int, t_peaks))
        }
    
    def _detect_peaks_custom(self, ecg_clean: np.ndarray) -> Dict[str, List[int]]:
        """Custom peak detection when NeuroKit2 is not available"""
        # Detect R-peaks using scipy
        distance = int(0.5 * self.sampling_rate)  # Minimum 0.5s between beats
        height = 0.3 * np.max(ecg_clean)
        
        r_peaks, _ = find_peaks(ecg_clean, distance=distance, height=height)
        
        if len(r_peaks) == 0:
            # Try with lower threshold
            height = 0.1 * np.max(ecg_clean)
            r_peaks, _ = find_peaks(ecg_clean, distance=distance, height=height)
        
        if len(r_peaks) == 0:
            return self._empty_peaks()
        
        # Estimate other peaks from R-peaks
        p_peaks, q_peaks, s_peaks, t_peaks = self._estimate_waves_from_r(r_peaks, len(ecg_clean))
        
        return {
            "p_peaks": list(map(int, p_peaks)),
            "q_peaks": list(map(int, q_peaks)),
            "r_peaks": list(map(int, r_peaks)),
            "s_peaks": list(map(int, s_peaks)),
            "t_peaks": list(map(int, t_peaks))
        }
    
    def _estimate_waves_from_r(self, r_peaks: np.ndarray, signal_length: int) -> Tuple[List, List, List, List]:
        """Estimate P, Q, S, T peaks based on typical timing relative to R-peaks"""
        p_peaks = []
        q_peaks = []
        s_peaks = []
        t_peaks = []
        
        for r in r_peaks:
            # P wave: ~160-200ms before R
            p = r - int(0.16 * self.sampling_rate)
            if p > 0:
                p_peaks.append(p)
            
            # Q wave: ~40ms before R
            q = r - int(0.04 * self.sampling_rate)
            if q > 0:
                q_peaks.append(q)
            
            # S wave: ~40ms after R
            s = r + int(0.04 * self.sampling_rate)
            if s < signal_length:
                s_peaks.append(s)
            
            # T wave: ~200-300ms after R
            t = r + int(0.25 * self.sampling_rate)
            if t < signal_length:
                t_peaks.append(t)
        
        return p_peaks, q_peaks, s_peaks, t_peaks
    
    def _clean_peaks(self, peaks) -> List[int]:
        """Remove NaN values and convert to integers"""
        if peaks is None:
            return []
        return [int(p) for p in peaks if not np.isnan(p)]
    
    def _empty_peaks(self) -> Dict[str, List[int]]:
        """Return empty peak dictionary"""
        return {
            "p_peaks": [],
            "q_peaks": [],
            "r_peaks": [],
            "s_peaks": [],
            "t_peaks": []
        }
    
    def calculate_heart_rate(self, r_peaks: List[int]) -> int:
        """
        Calculate heart rate from R-peaks
        
        Args:
            r_peaks: List of R-peak indices
            
        Returns:
            Heart rate in BPM
        """
        if len(r_peaks) < 2:
            return 0
        
        # Calculate R-R intervals in samples
        rr_intervals = np.diff(r_peaks)
        
        # Convert to seconds
        rr_seconds = rr_intervals / self.sampling_rate
        
        # Calculate average heart rate
        avg_rr = np.mean(rr_seconds)
        if avg_rr > 0:
            heart_rate = 60.0 / avg_rr
            return int(round(heart_rate))
        
        return 0
    
    def get_rr_variability(self, r_peaks: List[int]) -> float:
        """
        Calculate R-R interval variability (coefficient of variation)
        
        Args:
            r_peaks: List of R-peak indices
            
        Returns:
            Coefficient of variation (0-1 scale)
        """
        if len(r_peaks) < 3:
            return 0.0
        
        rr_intervals = np.diff(r_peaks)
        mean_rr = np.mean(rr_intervals)
        
        if mean_rr > 0:
            return float(np.std(rr_intervals) / mean_rr)
        
        return 0.0
    
    def analyze(self, ecg_signal: List[float]) -> Dict:
        """
        Complete ECG analysis
        
        Args:
            ecg_signal: Raw ECG signal values
            
        Returns:
            Dictionary with peaks, heart rate, and signal quality info
        """
        ecg_array = np.array(ecg_signal, dtype=np.float64)
        
        # Detect all peaks
        peaks = self.detect_peaks(ecg_array)
        
        # Calculate heart rate
        heart_rate = self.calculate_heart_rate(peaks["r_peaks"])
        
        # Calculate variability
        rr_variability = self.get_rr_variability(peaks["r_peaks"])
        
        return {
            **peaks,
            "heart_rate": heart_rate,
            "rr_variability": round(rr_variability, 3),
            "num_beats": len(peaks["r_peaks"]),
            "signal_length": len(ecg_array),
            "duration_seconds": round(len(ecg_array) / self.sampling_rate, 2)
        }
