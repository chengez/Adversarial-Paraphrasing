from .models.fast_detectgpt.fast_detectgpt import FastDetectGPT
from .models.gltr.gltr import GLTR



class Detector:
    """Shared interface for all detectors"""

    def inference(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list of scores from 0 to 1 with
        0 indicating likely human-written, and 1 indicating likely machine-generated."""
        pass


def get_detector(detector_name: str) -> Detector:
    
    if detector_name == "fastdetectgpt":
        return FastDetectGPT()
    elif detector_name == "gltr":
        return GLTR()
    else:
        raise ValueError("Invalid detector name")
