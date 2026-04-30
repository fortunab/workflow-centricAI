from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class ClassificationResult:
    label: str
    confidence: float
    logits: Optional[List[float]] = None

@dataclass
class DetectionBox:
    xyxy: List[float]
    confidence: float
    class_id: int = 0

@dataclass
class DetectionResult:
    boxes: List[DetectionBox] = field(default_factory=list)

@dataclass
class SegmentationResult:
    masks: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

class DetectionModule:
    def predict(self, image):
        return DetectionResult(boxes=[])

class SegmentationModule:
    def predict(self, image, boxes=None):
        return SegmentationResult(masks=[], scores=[])

class ReasoningModule:
    def explain(self, classification, detections, segmentations):
        return {
            "summary": f"Predicted {classification.label} with confidence {classification.confidence:.4f}.",
            "num_boxes": len(detections.boxes),
            "num_masks": len(segmentations.masks),
        }

class MedicalOrchestrator:
    def __init__(self, detector=None, segmenter=None, reasoner=None):
        self.detector = detector or DetectionModule()
        self.segmenter = segmenter or SegmentationModule()
        self.reasoner = reasoner or ReasoningModule()
    def run(self, image, classification):
        detections = self.detector.predict(image)
        segmentations = self.segmenter.predict(image, detections.boxes)
        rationale = self.reasoner.explain(classification, detections, segmentations)
        return {
            "classification": classification,
            "detections": detections,
            "segmentations": segmentations,
            "reasoning": rationale,
        }
