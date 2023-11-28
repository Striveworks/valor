# API

http://localhost:8000/docs#/

## **POST** `/groundtruths`
## **GET** `/groundtruths/dataset/{dataset_name}/datum/{uid}`
## **POST** `/predictions`
## **GET** `/predictions/model/{model_name}/dataset/{dataset_name}/datum/{uid}`
## **GET** `/labels`
## **GET** `/labels/dataset/{dataset_name}`
## **GET** `/labels/model/{model_name}`
## **POST** `/datasets`
## **GET** `/datasets`
## **GET** `/datasets/{dataset_name}`
## **GET** `/datasets/{dataset_name}/status`
## **PUT** `/datasets/{dataset_name}/finalize`
## **DELETE** `/datasets/{dataset_name}`
## **GET** `/data/dataset/{dataset_name}`
## **GET** `/data/dataset/{dataset_name}/uid/{uid}`
## **POST** `/models`
## **GET** `/models`
## **GET** `/models/{model_name}`
## **PUT** `/models/{model_name}/datasets/{dataset_name}/finalize`
## **DELETE** `/models/{model_name}`
## **POST** `/evaluations/ap-metrics`
## **POST** `/evaluations/clf-metrics`
## **POST** `/evaluations/semantic-segmentation-metrics`
## **GET** `/evaluations/datasets/{dataset_name}`
## **GET** `/evaluations/models/{model_name}`
## **GET** `/evaluations/`
## **GET** `/evaluations/{job_id}`
## **GET** `/evaluations/{job_id}/settings`
## **GET** `/evaluations/{job_id}/metrics`
## **GET** `/user`