# Velour documentation

_velour_ is an evaluation store that facilitates the computation, discoverability, and shareability of metrics for meachine learning models.

The core of _velour_ is a backend REST API service. user's will typically interact with this service via a python client. there is also a lightweight web interface. At a high-level, the typical workflow involves posting groundtruth annotations (class labels, bounding boxes, segmentation masks, etc.) and model predictions to the service. Velour, on the backend, then handles the computation of metrics, stores them centrally, and allows them to be queried. Velour does _not_ store raw data (such as underlying images) or facilitate model inference. It only stores interacts with groundtruth annotations and the predictions outputted from a model.
