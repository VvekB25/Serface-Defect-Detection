from ml.config.settings import MODEL_PATH
from ml.inference.model_loader import load_trained_model
from ml.training.data_loader import create_generators


def evaluate_model():
    _, validation_generator = create_generators()
    model = load_trained_model(MODEL_PATH)
    return model.evaluate(validation_generator, return_dict=True)


if __name__ == "__main__":
    print(evaluate_model())
