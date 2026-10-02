import unittest
from pathlib import Path
import tempfile

from PIL import Image

from ml.config.settings import CLASS_NAMES_PATH, MODEL_PATH
from ml.inference.predictor import DefectPredictor


class InferencePipelineTest(unittest.TestCase):
    def test_load_preprocess_predict_and_validate_result(self):
        if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
            self.skipTest(
                "Inference artifacts are missing. Train the model before running this test."
            )

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            Image.new("RGB", (128, 128), color=(128, 128, 128)).save(image_path)

            predictor = DefectPredictor()
            result = predictor.predict(image_path)

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIsInstance(result["predicted_class"], str)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
