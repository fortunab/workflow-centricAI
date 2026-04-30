import numpy as np
from project.pipelines.medical_orchestrator import MedicalOrchestrator, ClassificationResult
from project.evaluation.reporting import save_pipeline_summary
from project.configs.settings import OUTPUTS_DIR

def main():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    classification = ClassificationResult(label="polyp", confidence=0.9132)
    orchestrator = MedicalOrchestrator()
    payload = orchestrator.run(image, classification)
    save_pipeline_summary(payload, OUTPUTS_DIR / "orchestrator")

if __name__ == "__main__":
    main()
