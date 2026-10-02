# Phase 1: ML Bifurcation

The original notebook is a six-class steel surface defect classifier based on a custom TensorFlow/Keras CNN. It converts XML annotations into class directories, trains with image augmentation, evaluates on a validation generator, saves a model, and predicts one uploaded image.

## Training boundary

Training-only code includes Google Drive mounting, XML conversion, class-directory creation, augmentation, directory generators, model compilation, `model.fit`, training history, and plots. These are represented under `ml/preprocessing`, `ml/training`, and `ml/evaluation`.

## Production inference boundary

Inference requires only the saved Keras model, the class mapping, image resizing to 128x128, RGB conversion, division by 255, prediction, argmax selection, and confidence formatting. These are represented under `ml/inference` and `ml/preprocessing/image_preprocessor.py`.

## Important findings

- The notebook creates converted data under `/content/dataset`, but its later generator paths point to the raw image directories. The extracted loader uses `data/processed/train` and `data/processed/validation`, while the raw dataset is configured at the project-root `NEU-DET` directory.
- The class name `rolled_in_scale` must be used consistently. The notebook prediction cell uses `rolled-in_scale`.
- The source XML annotations use `rolled-in_scale`; dataset conversion normalizes that source alias to the canonical `rolled_in_scale` class folder and output label.
- The XML converter reads only the first `object` and ignores bounding-box coordinates. This project is therefore classification, not object detection.
- No annotated image or defect coordinates are currently produced.
- The saved model and class mapping are the only model artifacts required for inference.
- Backend, frontend, database, authentication, and deployment are intentionally outside Phase 1.
