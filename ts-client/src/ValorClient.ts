import axios, { AxiosInstance } from 'axios';

type Dataset = {
  id: number;
  name: string;
  metadata: object;
};

type Model = {
  id: number;
  name: string;
  metadata: object;
};

type TaskType =
  | 'skip'
  | 'empty'
  | 'classification'
  | 'object-detection'
  | 'semantic-segmentation'
  | 'embedding';

type Evaluation = {
  id: number;
  model_name: string;
  datum_filter: object;
  parameters: { task_type: TaskType; object };
  status: 'pending' | 'running' | 'done' | 'failed' | 'deleting';
  metrics: object[];
  confusion_matrices: object[];
};

const metadataDictToString = (input: { [key: string]: string | number }): string => {
  const result: { [key: string]: Array<{ value: string | number; operator: string }> } =
    {};

  Object.entries(input).forEach(([key, value]) => {
    result[key] = [{ value: value, operator: '==' }];
  });

  return JSON.stringify(result);
};

export class ValorClient {
  private client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  public async getDatasets(queryParams: object): Promise<Dataset[]> {
    const response = await this.client.get('/datasets', { params: queryParams });
    return response.data;
  }

  public async getDatasetsByMetadata(metadata: {
    [key: string]: string | number;
  }): Promise<Dataset[]> {
    return this.getDatasets({ dataset_metadata: metadataDictToString(metadata) });
  }

  public async getDatasetByName(name: string): Promise<Dataset> {
    const response = await this.client.get(`/datasets/${name}`);
    return response.data;
  }

  public async createDataset(name: string, metadata: object): Promise<void> {
    await this.client.post('/datasets', { name, metadata });
  }

  public async finalizeDataset(name: string): Promise<void> {
    await this.client.put(`/datasets/${name}/finalize`);
  }

  public async deleteDataset(name: string): Promise<void> {
    await this.client.delete(`/datasets/${name}`);
  }

  public async getModels(queryParams: object): Promise<Model[]> {
    const response = await this.client.get('/models', { params: queryParams });
    return response.data;
  }

  public async getModelsByMetadata(metadata: {
    [key: string]: string | number;
  }): Promise<Model[]> {
    return this.getModels({ model_metadata: metadataDictToString(metadata) });
  }

  public async getModelByName(name: string): Promise<Model> {
    const response = await this.client.get(`/models/${name}`);
    return response.data;
  }

  public async createModel(name: string, metadata: object): Promise<void> {
    await this.client.post('/models', { name, metadata });
  }

  public async deleteModel(name: string): Promise<void> {
    await this.client.delete(`/models/${name}`);
  }

  public async createOrGetEvaluation(
    model: string,
    dataset: string,
    taskType: TaskType
  ): Promise<Evaluation> {
    const response = await this.client.post('/evaluations', {
      model_names: [model],
      datum_filter: { dataset_names: [dataset] },
      parameters: { task_type: taskType }
    });
    return response.data[0];
  }

  public async getEvaluations(queryParams: object): Promise<Evaluation[]> {
    const response = await this.client.get('/evaluations', { params: queryParams });
    return response.data;
  }

  public async getEvaluationById(id: number): Promise<Evaluation> {
    const evaluations = await this.getEvaluations({ evaluation_ids: id });
    return evaluations[0];
  }

  public async getEvaluationsByModelName(modelName: string): Promise<Evaluation[]> {
    return this.getEvaluations({ models: modelName });
  }

  public async getEvaluationsByDatasetName(datasetName: string): Promise<Evaluation[]> {
    return this.getEvaluations({ datasets: datasetName });
  }

  public async addGroundTruth(
    datasetName: string,
    datumUid: string,
    annotations: object[]
  ): Promise<void> {
    await this.client.post('/groundtruths', [
      { datum: { uid: datumUid, dataset_name: datasetName }, annotations: annotations }
    ]);
  }

  public async addPredictions(
    modelName: string,
    datasetName: string,
    datumUid: string,
    annotations: object[]
  ): Promise<void> {
    await this.client.post('/predictions', [
      {
        model_name: modelName,
        datum: { uid: datumUid, dataset_name: datasetName },
        annotations: annotations
      }
    ]);
  }
}
