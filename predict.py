import argparse

from ml.inference.predictor import DefectPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a steel surface defect from an image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", help="Optional path to a .keras model")
    parser.add_argument("--classes", help="Optional path to the class mapping JSON")
    args = parser.parse_args()

    predictor_kwargs = {}
    if args.model:
        predictor_kwargs["model_path"] = args.model
    if args.classes:
        predictor_kwargs["class_names_path"] = args.classes
    predictor = DefectPredictor(**predictor_kwargs)
    result = predictor.predict(args.image)

    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
