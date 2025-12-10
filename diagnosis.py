"""
ECG Diagnosis Module
Rule-based abnormality detection and diagnosis generation
"""

from typing import Dict, List, Tuple


class ECGDiagnostics:
    """Rule-based ECG abnormality detection and diagnosis"""
    
    # Heart rate thresholds
    HR_BRADYCARDIA = 60
    HR_TACHYCARDIA = 100
    HR_SEVERE_BRADY = 40
    HR_SEVERE_TACHY = 150
    
    # R-R variability thresholds
    RR_IRREGULAR_THRESHOLD = 0.15  # 15% coefficient of variation
    RR_HIGHLY_IRREGULAR = 0.25
    
    def __init__(self):
        pass
    
    def analyze(self, analysis_data: Dict) -> Tuple[List[str], str]:
        """
        Analyze ECG data and generate diagnosis
        
        Args:
            analysis_data: Dictionary from ECGProcessor.analyze()
            
        Returns:
            Tuple of (abnormalities list, final diagnosis string)
        """
        abnormalities = []
        warnings = []
        
        heart_rate = analysis_data.get("heart_rate", 0)
        rr_variability = analysis_data.get("rr_variability", 0)
        num_beats = analysis_data.get("num_beats", 0)
        r_peaks = analysis_data.get("r_peaks", [])
        p_peaks = analysis_data.get("p_peaks", [])
        
        # Check signal quality
        if num_beats == 0:
            return (
                ["No heartbeats detected", "Signal quality issue"],
                "Unable to analyze ECG. No heartbeats detected. Please check signal quality."
            )
        
        if num_beats < 3:
            warnings.append("Limited data for accurate analysis")
        
        # === Heart Rate Analysis ===
        hr_status, hr_abnormality = self._analyze_heart_rate(heart_rate)
        if hr_abnormality:
            abnormalities.append(hr_abnormality)
        
        # === Rhythm Analysis ===
        rhythm_status, rhythm_abnormalities = self._analyze_rhythm(
            rr_variability, r_peaks, heart_rate
        )
        abnormalities.extend(rhythm_abnormalities)
        
        # === Wave Morphology Analysis ===
        morphology_abnormalities = self._analyze_morphology(
            p_peaks, r_peaks, analysis_data
        )
        abnormalities.extend(morphology_abnormalities)
        
        # === Generate Final Diagnosis ===
        if not abnormalities:
            abnormalities = ["Normal Sinus Rhythm", "No Arrhythmia Detected"]
            final_diagnosis = self._generate_normal_diagnosis(heart_rate)
        else:
            final_diagnosis = self._generate_abnormal_diagnosis(
                abnormalities, heart_rate, warnings
            )
        
        return abnormalities, final_diagnosis
    
    def _analyze_heart_rate(self, heart_rate: int) -> Tuple[str, str]:
        """Analyze heart rate and return status and any abnormality"""
        if heart_rate == 0:
            return "unknown", "Unable to determine heart rate"
        
        if heart_rate < self.HR_SEVERE_BRADY:
            return "severe_bradycardia", f"Severe Bradycardia ({heart_rate} BPM)"
        
        if heart_rate < self.HR_BRADYCARDIA:
            return "bradycardia", f"Bradycardia ({heart_rate} BPM)"
        
        if heart_rate > self.HR_SEVERE_TACHY:
            return "severe_tachycardia", f"Severe Tachycardia ({heart_rate} BPM)"
        
        if heart_rate > self.HR_TACHYCARDIA:
            return "tachycardia", f"Tachycardia ({heart_rate} BPM)"
        
        return "normal", ""
    
    def _analyze_rhythm(
        self, 
        rr_variability: float, 
        r_peaks: List[int],
        heart_rate: int
    ) -> Tuple[str, List[str]]:
        """Analyze rhythm regularity"""
        abnormalities = []
        
        if len(r_peaks) < 3:
            return "insufficient_data", []
        
        # Check R-R variability
        if rr_variability > self.RR_HIGHLY_IRREGULAR:
            abnormalities.append("Highly Irregular Rhythm")
            abnormalities.append("Possible Atrial Fibrillation")
            return "irregular", abnormalities
        
        if rr_variability > self.RR_IRREGULAR_THRESHOLD:
            abnormalities.append("Irregular Rhythm")
            abnormalities.append("Possible Sinus Arrhythmia")
            return "mildly_irregular", abnormalities
        
        return "regular", abnormalities
    
    def _analyze_morphology(
        self, 
        p_peaks: List[int], 
        r_peaks: List[int],
        analysis_data: Dict
    ) -> List[str]:
        """Analyze wave morphology for abnormalities"""
        abnormalities = []
        
        if len(r_peaks) == 0:
            return abnormalities
        
        # Check P-wave presence (should be roughly 1:1 with R-waves in normal rhythm)
        if len(p_peaks) > 0 and len(r_peaks) > 0:
            p_to_r_ratio = len(p_peaks) / len(r_peaks)
            
            if p_to_r_ratio < 0.5:
                abnormalities.append("Missing P-waves detected")
                abnormalities.append("Possible Junctional Rhythm")
            elif p_to_r_ratio > 1.5:
                abnormalities.append("Extra P-waves detected")
                abnormalities.append("Possible AV Block")
        
        return abnormalities
    
    def _generate_normal_diagnosis(self, heart_rate: int) -> str:
        """Generate diagnosis for normal ECG"""
        return (
            f"Normal ECG. Heart rate {heart_rate} BPM within normal range. "
            f"Regular sinus rhythm with no significant abnormalities detected."
        )
    
    def _generate_abnormal_diagnosis(
        self, 
        abnormalities: List[str], 
        heart_rate: int,
        warnings: List[str]
    ) -> str:
        """Generate diagnosis for abnormal ECG"""
        # Determine severity
        severe_conditions = [
            "Severe Bradycardia", "Severe Tachycardia", 
            "Possible Atrial Fibrillation", "Possible AV Block"
        ]
        
        is_severe = any(
            cond in abnormality 
            for abnormality in abnormalities 
            for cond in severe_conditions
        )
        
        # Build diagnosis
        diagnosis_parts = []
        
        if is_severe:
            diagnosis_parts.append("ATTENTION REQUIRED:")
        
        diagnosis_parts.append(f"Heart rate: {heart_rate} BPM.")
        
        # Summarize findings
        if "Bradycardia" in str(abnormalities):
            diagnosis_parts.append("Heart rate below normal range.")
        elif "Tachycardia" in str(abnormalities):
            diagnosis_parts.append("Heart rate above normal range.")
        
        if "Irregular" in str(abnormalities):
            diagnosis_parts.append("Irregular rhythm pattern detected.")
        
        if "Atrial Fibrillation" in str(abnormalities):
            diagnosis_parts.append("Pattern consistent with atrial fibrillation.")
        
        # Add recommendation
        if is_severe:
            diagnosis_parts.append("Medical consultation recommended.")
        else:
            diagnosis_parts.append("Monitor for changes.")
        
        if warnings:
            diagnosis_parts.append(f"Note: {'; '.join(warnings)}")
        
        return " ".join(diagnosis_parts)


def get_diagnosis(analysis_data: Dict) -> Dict:
    """
    Convenience function to get complete diagnosis
    
    Args:
        analysis_data: Dictionary from ECGProcessor.analyze()
        
    Returns:
        Dictionary with abnormalities and final_diagnosis added
    """
    diagnostics = ECGDiagnostics()
    abnormalities, final_diagnosis = diagnostics.analyze(analysis_data)
    
    return {
        **analysis_data,
        "abnormalities": abnormalities,
        "final_diagnosis": final_diagnosis
    }
