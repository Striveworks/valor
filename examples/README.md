# Examples

This folder contains various examples of Velour usage.

| File | Description |
| --- | --- |
| [getting_started.ipynb](getting_started.ipynb) | A Jupyter notebook that walks through the basics of using Velour. ***This is a good place to start!*** |
| [pedestrian_detection.ipynb](pedestrian-detection.ipynb) | A Jupyter notebook that walks through an object detection example, showing how to use the power of Velour's filtering functionality to provide a fine-grained analysis of model performance with respect to user defined business logic. ***This is a good place to go after `getting_started.ipynb`*** |
| [tabular_classification.ipynb](tabular_classification.ipynb) | A Jupyter notebook showing an end-to-end example of evaluating a scikit-learn classification model. |
| [detection](detection) | This folder demonstrates both how to evaluate an object detection model and provides example scripts of how to integrate models and datasets into Velour. `integrations/coco_integration.py` demonstrates (using the COCO dataset as an example) the type of integration code necessary to integrate existing annotations into Velour, while `yolo_integration.py` demonstrates (using the Ultralytics YOLO model as an example) the type of integration code necessary to integrate model outputs into Velour. The notebook `coco-yolo.ipynb` shows, using the integration scripts, how to evaluate an object detection model.  |