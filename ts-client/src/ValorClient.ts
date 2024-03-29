import axios, { AxiosInstance } from 'axios';

export type Dataset = {
  name: string;
  metadata: Partial<Record<string, any>>;
};

export type Model = {
  name: string;
  metadata: Partial<Record<string, any>>;
};

export type TaskType =
  | 'skip'
  | 'empty'
  | 'classification'
  | 'object-detection'
  | 'semantic-segmentation'
  | 'embedding';

export type Label = {
  key: string;
  value: string;
  score?: number;
};

export type Metric = {
  type: string;
  parameters?: Partial<Record<string, any>>;
  value: number | any;
  labe?: Label;
};

export type Evaluation = {
  id: number;
  model_name: string;
  datum_filter: { dataset_names: string[]; object: any };
  parameters: { task_type: TaskType; object: any };
  status: 'pending' | 'running' | 'done' | 'failed' | 'deleting';
  metrics: Metric[];
  confusion_matrices: any[];
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

  /**
   *
   * @param baseURL - The base URL of the Valor server to connect to.
   */
  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Fetches datasets matching the filters defined by queryParams. This is private
   * because we define higher-level methods that use this.
   *
   * @param queryParams An object containing query parameters to filter datasets by.
   *
   * @returns {Promise<Dataset[]>}
   *
   */
  private async getDatasets(queryParams: object): Promise<Dataset[]> {
    const response = await this.client.get('/datasets', { params: queryParams });
    return response.data;
  }

  /**
   * Fetches all datasets
   *
   * @returns {Promise<Dataset[]>}
   */
  public async getAllDatasets(): Promise<Dataset[]> {
    return this.getDatasets({});
  }

  /**
   * Fetches datasets matching a metadata object
   *
   * @param {{[key: string]: string | number}} metadata A metadata object to filter datasets by.
   *
   * @returns {Promise<Dataset[]>}
   *
   * @example
   * const client = new ValorClient('http://localhost:8000/');
   * client.getDatasetsByMetadata({ some_key: some_value }) // returns all datasets that have a metadata field `some_key` with value `some_value`
   *
   */
  public async getDatasetsByMetadata(metadata: {
    [key: string]: string | number;
  }): Promise<Dataset[]> {
    return this.getDatasets({ dataset_metadata: metadataDictToString(metadata) });
  }

  /**
   * Fetches a dataset given its name
   *
   * @param name name of the dataset
   *
   * @returns {Promise<Dataset>}
   */
  public async getDatasetByName(name: string): Promise<Dataset> {
    const response = await this.client.get(`/datasets/${name}`);
    return response.data;
  }

  /**
   * Creates a new dataset
   *
   * @param name name of the dataset
   * @param metadata metadata of the dataset
   *
   * @returns {Promise<void>}
   */
  public async createDataset(name: string, metadata: object): Promise<void> {
    await this.client.post('/datasets', { name, metadata });
  }

  /**
   * Finalizes a dataset (which is necessary to run an evaluation)
   *
   * @param name name of the dataset to finalize
   *
   * @returns {Promise<void>}
   */
  public async finalizeDataset(name: string): Promise<void> {
    await this.client.put(`/datasets/${name}/finalize`);
  }

  /**
   * Deletes a dataset
   *
   * @param name name of the dataset to delete
   *
   * @returns {Promise<void>}
   */
  public async deleteDataset(name: string): Promise<void> {
    await this.client.delete(`/datasets/${name}`);
  }

  /**
   * Fetches models matching the filters defined by queryParams. This is
   * private because we define higher-level methods that use this.
   *
   * @param queryParams An object containing query parameters to filter models by.
   *
   * @returns {Promise<Model[]>}
   */
  private async getModels(queryParams: object): Promise<Model[]> {
    const response = await this.client.get('/models', { params: queryParams });
    return response.data;
  }

  /**
   * Fetches all models
   *
   * @returns {Promise<Model[]>}
   */
  public async getAllModels(): Promise<Model[]> {
    return this.getModels({});
  }

  /**
   * Fetches models matching a metadata object
   *
   * @param {{[key: string]: string | number}} metadata A metadata object to filter models by.
   *
   * @returns {Promise<Model[]>}
   *
   * @example
   * const client = new ValorClient('http://localhost:8000/');
   * client.getModelsByMetadata({ some_key: some_value }) // returns all models that have a metadata field `some_key` with value `some_value`
   */
  public async getModelsByMetadata(metadata: {
    [key: string]: string | number;
  }): Promise<Model[]> {
    return this.getModels({ model_metadata: metadataDictToString(metadata) });
  }

  /**
   * Fetches a model given its name
   *
   * @param name name of the model
   *
   * @returns {Promise<Model>}
   */
  public async getModelByName(name: string): Promise<Model> {
    const response = await this.client.get(`/models/${name}`);
    return response.data;
  }

  /**
   * Creates a new model
   *
   * @param name name of the model
   * @param metadata metadata of the model
   *
   * @returns {Promise<void>}
   */
  public async createModel(name: string, metadata: object): Promise<void> {
    await this.client.post('/models', { name, metadata });
  }

  /**
   * Deletes a model
   *
   * @param name name of the model to delete
   *
   * @returns {Promise<void>}
   */
  public async deleteModel(name: string): Promise<void> {
    await this.client.delete(`/models/${name}`);
  }

  /**
   * Creates a new evaluation or gets an existing one if an evaluation with the
   * same parameters already exists.
   *
   * @param model name of the model
   * @param dataset name of the dataset
   * @param taskType type of task
   *
   * @returns {Promise<Evaluation>}
   */
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

  /**
   * Creates new evaluations given a list of models, or gets existing ones if evaluations with the
   * same parameters already exists.
   *
   * @param models names of the models
   * @param dataset name of the dataset
   * @param taskType type of task
   *
   * @returns {Promise<Evaluation[]>}
   */
  public async bulkCreateOrGetEvaluations(
    models: string[],
    dataset: string,
    taskType: TaskType
  ): Promise<Evaluation[]> {
    const response = await this.client.post('/evaluations', {
      model_names: models,
      datum_filter: { dataset_names: [dataset] },
      parameters: { task_type: taskType }
    });
    return response.data;
  }

  /**
   * Fetches evaluations matching the filters defined by queryParams. This is
   * private because we define higher-level methods that use this.
   *
   * @param queryParams An object containing query parameters to filter evaluations by.
   *
   * @returns {Promise<Evaluation[]>}
   */
  private async getEvaluations(queryParams: object): Promise<Evaluation[]> {
    const response = await this.client.get('/evaluations', { params: queryParams });
    return response.data;
  }

  /**
   * Fetches an evaluation by id
   *
   * @param id id of the evaluation
   *
   * @returns {Promise<Evaluation>}
   */
  public async getEvaluationById(id: number): Promise<Evaluation> {
    const evaluations = await this.getEvaluations({ evaluation_ids: id });
    return evaluations[0];
  }

  /**
   * Bulk fetches evaluation by array of ids
   *
   * @param id id of the evaluation
   *
   * @returns {Promise<Evaluation[]>}
   */
  public async getEvaluationsByIds(ids: number[]): Promise<Evaluation[]> {
    const evaluations = await this.getEvaluations({
      evaluation_ids: ids.map((id) => id.toString()).join(',')
    });
    return evaluations;
  }

  /**
   * Fetches all evaluations associated to given models
   *
   * @param modelNames names of the models
   *
   * @returns {Promise<Evaluation[]>}
   */
  public async getEvaluationsByModelNames(modelNames: string[]): Promise<Evaluation[]> {
    // turn modelNames into a comma-separated string
    return this.getEvaluations({ models: modelNames.join(',') });
  }

  /**
   * Fetches all evaluations associated to given datasets
   *
   * @param datasetNames names of the datasets
   *
   * @returns {Promise<Evaluation[]>}
   */
  public async getEvaluationsByDatasetNames(
    datasetNames: string[]
  ): Promise<Evaluation[]> {
    return this.getEvaluations({ datasets: datasetNames.join(',') });
  }

  /**
   * Adds ground truth annotations to a dataset
   *
   * @param datasetName name of the dataset
   * @param datumUid uid of the datum
   * @param annotations annotations to add
   *
   * @returns {Promise<void>}
   */
  public async addGroundTruth(
    datasetName: string,
    datumUid: string,
    annotations: object[]
  ): Promise<void> {
    return this.client.post('/groundtruths', [
      { datum: { uid: datumUid, dataset_name: datasetName }, annotations: annotations }
    ]);
  }

  /**
   * Adds predictions from a model
   *
   * @param modelName name of the model
   * @param datasetName name of the dataset
   * @param datumUid uid of the datum
   * @param annotations annotations to add
   *
   * @returns {Promise<void>}
   */
  public async addPredictions(
    modelName: string,
    datasetName: string,
    datumUid: string,
    annotations: object[]
  ): Promise<void> {
    return this.client.post('/predictions', [
      {
        model_name: modelName,
        datum: { uid: datumUid, dataset_name: datasetName },
        annotations: annotations
      }
    ]);
  }
}
