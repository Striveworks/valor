import requests
import gzip
import json
import tempfile
import pathlib

from chariot.datasets.dataset_version import DatasetVersion

from velour.data_types import (
    BoundingBox,
    BoundingPolygon,
    GroundTruthDetection,
    GroundTruthImageClassification,
    GroundTruthInstanceSegmentation,
    GroundTruthSemanticSegmentation,
    Image,
    Label,
    Point,
    PolygonWithHole,
    PredictedDetection,
    PredictedImageClassification,
    PredictedInstanceSegmentation,
    ScoredLabel,
)

def retrieve_chariot_ds(manifest_url: str):
    """Retrieves and unpacks Chariot dataset annotations from a manifest url."""

    chariot_dataset = []  

    # Create a temporary file
    with tempfile.TemporaryFile(mode='w+b') as f:

        # Download compressed jsonl file
        response = requests.get(manifest_url, stream=True)
        if response.status_code == 200:
            f.write(response.raw.read())
        f.flush()
        f.seek(0)

        # Unzip 
        gzf = gzip.GzipFile(mode='rb', fileobj=f)        
        jsonstr = gzf.read().decode().strip().split('\n')

        # Parse into list of json object(s)
        for line in jsonstr:
            try:
                chariot_dataset.append(json.loads(line))                
            except:
                pass
    
    return chariot_dataset
    
def parse_chariot_image_classification_annotation(datum: dict) -> GroundTruthImageClassification:
    """Parses Chariot image classification annotation."""

    # Strip UID from URL path
    uid = pathlib.Path(datum['path']).stem

    gt_dets = []
    for annotation in datum['annotations']:
        gt_dets.append(GroundTruthImageClassification(
            image=Image(uid=uid, height=None, width=None),
            labels=[Label(key='class_label', value=annotation['class_label'])],
        ))
    return gt_dets

def parse_chariot_image_segmentation_annotation(datum: dict) -> GroundTruthSemanticSegmentation:
    """Parses Chariot image segmentation annotation."""

    # Strip UID from URL path
    uid = pathlib.Path(datum['path']).stem

    annotated_regions = {}    
    for annotation in datum['annotations']:

        annotation_label = annotation['class_label']

        # Create PolygonWithHole
        region = None
        if 'contours' in annotation:
            polygon = None
            hole = None

            # Create Bounding Polygon
            points = []
            for point in annotation['contours'][0]:
                points.append(Point(x=annotation['contours'][0]['x'], y=annotation['contours'][0]['y']))
            polygon = BoundingPolygon(points)

            # Check if hole exists
            if len(annotation['contours'] > 1):
                # Create Bounding Polygon for Hole
                points = []
                for point in annotation['contours'][1]:
                    points.append(Point(x=annotation['contours'][1]['x'], y=annotation['contours'][1]['y']))
                hole = BoundingPolygon(points)
            
            # Populate PolygonWithHole
            region = PolygonWithHole(polygon=polygon, hole=hole)
        
        # Add annotated region to list
        if region is not None:
            if annotation_label not in annotated_regions:
                annotated_regions[annotation_label] = []
            annotated_regions[annotation_label].append(region)

    # Create a list of GroundTruths
    gt_dets = []
    for label in annotated_regions.keys():
        gt_dets.append(GroundTruthSemanticSegmentation(
            shape=annotated_regions[label],
            labels=[Label(key='class_label', value=label)],
            image=Image(uid=uid, height=None, width=None),
        ))
    return gt_dets

def parse_chariot_object_detection_annotation(datum: dict) -> GroundTruthDetection:
    """Parses Chariot object detection annotation."""

    # Strip UID from URL path
    uid = pathlib.Path(datum['path']).stem
    
    gt_dets = []
    for annotation in datum['annotations']:
        gt_dets.append(GroundTruthDetection(
            bbox=BoundingBox(xmin=annotation['bbox']['xmin'], ymin=annotation['bbox']['ymin'], xmax=annotation['bbox']['xmax'], ymax=annotation['bbox']['ymax']),
            labels=[Label(key='class_label', value=annotation['class_label'])],
            image=Image(uid=uid, height=None, width=None),
        ))
    return gt_dets

def parse_chariot_text_sentiment_annotation(datum: dict):
    """Parses Chariot text sentiment annotation."""
    return None

def parse_chariot_text_summarization_annotation(datum: dict):
    """Parses Chariot text summarization annotation."""
    return None

def parse_chariot_text_token_classification_annotation(datum: dict):
    """Parses Chariot text token classification annotation."""
    return None

def parse_chariot_text_translation_annotation(datum: dict):
    """Parses Chariot text translation annotation."""
    return None

def chariot_ds_to_velour_ds(dsv: DatasetVersion, velour_name: str, use_training_manifest=True):
    """Converts the annotations of a Chariot dataset to velour's format.
    
    Parameters
    ----------
    dsv
        Chariot DatasetVerion object 
    velour_name
        Name of the new Velour dataset
    using_training_manifest
        (OPTIONAL) Defaults to true, setting false will use the evaluation manifest which is a 
        super set of the training manifest. Not recommended as the evaluation manifest may 
        contain unlabeled data.

    Returns
    -------
    List of Ground Truth Detections
    """
    
    manifest_url = dsv.get_training_manifest_url() if use_training_manifest else dsv.get_evaluation_manifest_url()
    supported_types = dsv.supported_task_types

    chariot_ds = retrieve_chariot_ds(manifest_url)
    gt_dets = []

    print("\nDType: " + str(dsv.supported_task_types))
    first_pass = True

    for datum in chariot_ds:        

        # Image Classification
        if dsv.supported_task_types.image_classification:                
            gt_dets += parse_chariot_image_classification_annotation(datum)

        # Image Segmentation
        if dsv.supported_task_types.image_segmentation:
            gt_dets += parse_chariot_image_segmentation_annotation(datum)

        # Object Detection
        if dsv.supported_task_types.object_detection:
            gt_dets += parse_chariot_object_detection_annotation(datum)

        # Text Sentiment
        if dsv.supported_task_types.text_sentiment:
            pass

        # Text Summarization
        if dsv.supported_task_types.text_summarization:
            pass

        # Text Token Classifier
        if dsv.supported_task_types.text_token_classification:
            pass

        # Text Translation
        if dsv.supported_task_types.text_translation:
            pass

        first_pass = False

    return gt_dets

### TESTING ###

def test_load():

    # List available datasets in project
    project_name="Global"
    # project_name="OnBoarding"
    datasets = chariot.datasets.dataset.get_datasets_in_project(limit=25, offset=0, project_name=project_name)

    dslu = {}
    print("Datasets")
    for i in range(len(datasets)):        
        dslu[str(datasets[i].name).strip()] = datasets[i]
        print(" " + str(i) + ": " + datasets[i].name)

    idx = int(input())
    dsv = datasets[idx].versions[0]

    retval = chariot_ds_to_velour_ds(dsv, "Test")
    
if __name__ == "__main__":

    import chariot
    from chariot.client import connect

    connect(host="https://production.chariot.striveworks.us")

    test_load()