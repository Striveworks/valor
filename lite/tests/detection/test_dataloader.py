import json

import pytest
from valor_lite.detection import DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()


def test_valor_integration():

    gt_json = '{"datum": {"uid": "139", "text": null, "metadata": {"license": 2, "file_name": "000000000139.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000139.jpg", "date_captured": "2013-11-21 01:34:01", "flickr_url": "http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg", "height": 426, "width": 640}}, "annotations": [{"metadata": {}, "labels": [{"key": "supercategory", "value": "person", "score": null}, {"key": "name", "value": "person", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[158.0, 413.0], [295.0, 413.0], [295.0, 465.0], [158.0, 465.0], [158.0, 413.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "person", "score": null}, {"key": "name", "value": "person", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[172.0, 384.0], [207.0, 384.0], [207.0, 399.0], [172.0, 399.0], [172.0, 384.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "chair", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[223.0, 413.0], [303.0, 413.0], [303.0, 442.0], [223.0, 442.0], [223.0, 413.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "chair", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[218.0, 291.0], [315.0, 291.0], [315.0, 352.0], [218.0, 352.0], [218.0, 291.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "chair", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[219.0, 412.0], [231.0, 412.0], [231.0, 421.0], [219.0, 421.0], [219.0, 412.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "chair", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[219.0, 317.0], [230.0, 317.0], [230.0, 338.0], [219.0, 338.0], [219.0, 317.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "chair", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[218.0, 359.0], [320.0, 359.0], [320.0, 414.0], [218.0, 414.0], [218.0, 359.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "potted plant", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[149.0, 237.0], [210.0, 237.0], [210.0, 260.0], [149.0, 260.0], [149.0, 237.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "furniture", "score": null}, {"key": "name", "value": "dining table", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[231.0, 321.0], [319.0, 321.0], [319.0, 446.0], [231.0, 446.0], [231.0, 321.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "electronic", "score": null}, {"key": "name", "value": "tv", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[168.0, 7.0], [262.0, 7.0], [262.0, 155.0], [168.0, 155.0], [168.0, 7.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "electronic", "score": null}, {"key": "name", "value": "tv", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[209.0, 557.0], [287.0, 557.0], [287.0, 638.0], [209.0, 638.0], [209.0, 557.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "appliance", "score": null}, {"key": "name", "value": "microwave", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[206.0, 512.0], [221.0, 512.0], [221.0, 526.0], [206.0, 526.0], [206.0, 512.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "appliance", "score": null}, {"key": "name", "value": "refrigerator", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[174.0, 493.0], [281.0, 493.0], [281.0, 512.0], [174.0, 512.0], [174.0, 493.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "book", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[308.0, 613.0], [353.0, 613.0], [353.0, 625.0], [308.0, 625.0], [308.0, 613.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "book", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[306.0, 605.0], [350.0, 605.0], [350.0, 618.0], [306.0, 618.0], [306.0, 605.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "clock", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[121.0, 448.0], [142.0, 448.0], [142.0, 461.0], [121.0, 461.0], [121.0, 448.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "vase", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[195.0, 241.0], [212.0, 241.0], [212.0, 254.0], [195.0, 254.0], [195.0, 241.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "vase", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[309.0, 549.0], [398.0, 549.0], [398.0, 584.0], [309.0, 584.0], [309.0, 549.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "vase", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[209.0, 351.0], [230.0, 351.0], [230.0, 361.0], [209.0, 361.0], [209.0, 351.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "supercategory", "value": "indoor", "score": null}, {"key": "name", "value": "vase", "score": null}, {"key": "iscrowd", "value": "0", "score": null}], "bounding_box": [[[200.0, 337.0], [215.0, 337.0], [215.0, 346.0], [200.0, 346.0], [200.0, 337.0]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}]}'
    pd_json = '{"datum": {"uid": "139", "text": null, "metadata": {"license": 2, "file_name": "000000000139.jpg", "coco_url": "http://images.cocodataset.org/val2017/000000000139.jpg", "date_captured": "2013-11-21 01:34:01", "flickr_url": "http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg", "height": 426, "width": 640}}, "annotations": [{"metadata": {}, "labels": [{"key": "name", "value": "tv", "score": 0.9257726073265076}, {"key": "unused_class", "value": "tv", "score": 0.9257726073265076}], "bounding_box": [[[4, 166], [155, 166], [155, 263], [4, 263], [4, 166]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "chair", "score": 0.866135835647583}], "bounding_box": [[[293, 217], [354, 217], [354, 319], [293, 319], [293, 217]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "chair", "score": 0.7706670761108398}], "bounding_box": [[[361, 217], [418, 217], [418, 310], [361, 310], [361, 217]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "person", "score": 0.7308055758476257}], "bounding_box": [[[416, 157], [465, 157], [465, 295], [416, 295], [416, 157]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "chair", "score": 0.6489511728286743}], "bounding_box": [[[405, 219], [444, 219], [444, 306], [405, 306], [405, 219]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "clock", "score": 0.6184478998184204}], "bounding_box": [[[448, 119], [461, 119], [461, 141], [448, 141], [448, 119]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "refrigerator", "score": 0.6119757294654846}], "bounding_box": [[[446, 167], [513, 167], [513, 289], [446, 289], [446, 167]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "potted plant", "score": 0.5597260594367981}], "bounding_box": [[[226, 178], [268, 178], [268, 212], [226, 212], [226, 178]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "vase", "score": 0.431998074054718}], "bounding_box": [[[550, 304], [585, 304], [585, 399], [550, 399], [550, 304]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "potted plant", "score": 0.3539217412471771}], "bounding_box": [[[334, 175], [370, 175], [370, 221], [334, 221], [334, 175]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "dining table", "score": 0.27812352776527405}], "bounding_box": [[[462, 350], [639, 350], [639, 423], [462, 423], [462, 350]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}, {"metadata": {}, "labels": [{"key": "name", "value": "tv", "score": 0.25976383686065674}], "bounding_box": [[[558, 207], [639, 207], [639, 296], [558, 296], [558, 207]]], "polygon": null, "raster": null, "embedding": null, "text": null, "context_list": null, "is_instance": true, "implied_task_types": null}]}'

    gt = json.loads(gt_json)
    pd = json.loads(pd_json)

    loader = DataLoader()
    loader.add_data_from_valor_dict([(gt, pd)])

    assert len(loader.pairs) == 1
    assert loader.pairs[0].shape == (281, 7)
