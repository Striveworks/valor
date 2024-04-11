import axios, { AxiosInstance } from 'axios';

type GeoJSONType = "Point" | "LineString" | "Polygon" | "MultiPoint" | "MultiLineString" | "MultiPolygon" | "GeometryCollection" | "Feature" | "FeatureCollection";

/**
 * Checks if value conforms to the GeoJSON specification.
 *
 * @param value The value to type check.
 * @returns A boolean result.
 */
function isGeoJSONObject(value: any): value is { type: GeoJSONType, coordinates: any } {
  const geoJSONTypes: GeoJSONType[] = ["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection", "Feature", "FeatureCollection"];
  return typeof value === 'object' && value !== null && 'type' in value && geoJSONTypes.includes(value.type as GeoJSONType);
}

/**
 * Encodes metadata into the Valor API format.
 *
 * @param input An object containing metadata.
 * @returns The encoded object.
 */
function encodeMetadata(input: { [key: string]: any }): { [key: string]: {type: string; value: any;} } {
  const output: { [key: string]: {type: string; value: any;} } = {};

  for (const key in input) {
    const value = input[key];
    let valueType: string;

    if (value instanceof Date) {
      valueType = 'datetime';
      output[key] = { type: valueType, value: value.toISOString() };
    } else if (isGeoJSONObject(value)) {
      valueType = 'geojson';
      output[key] = { type: valueType, value };
    } else if (typeof value === 'string') {
      valueType = 'string';
      output[key] = { type: valueType, value };
    } else if (typeof value === 'number') {
      valueType = Number.isInteger(value) ? 'integer' : 'float';
      output[key] = { type: valueType, value };
    } else {
      console.warn(`Unknown type for key "${key}".`);
      valueType = "unknown";
      output[key] = { type: valueType, value };
    }
  }

  return output;
}

/**
 * Decodes metadata from the Valor API format.
 *
 * @param input An encoded Valor metadata object.
 * @returns The decoded object.
 */
function decodeMetadata(input: { [key: string]: {type: string; value: any;} }): { [key: string]: any } {
  const output: { [key: string]: any } = {};

  for (const key in input) {
    const item = input[key];
    const { type, value } = item;

    switch (type) {
      case 'datetime':
      case 'date':
      case 'time':
        output[key] = new Date(value);
        break;
      case 'geojson':
        output[key] = value;
        break;
      case 'string':
      case 'integer':
      case 'float':
        output[key] = value;
        break;
      default:
        console.warn(`Unknown type for key "${key}".`);
        output[key] = value;
        break;
    }
  }

  return output;
}

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

export type Dataset = {
  name: string;
  metadata: Partial<Record<string, any>>;
};

export type Model = {
  name: string;
  metadata: Partial<Record<string, any>>;
};

export type Datum = {
  uid: string;
  metadata: Partial<Record<string, any>>;
}

export type Annotation = {
  task_type: TaskType;
  metadata: Partial<Record<string, any>>;
  labels: Label[];
  bounding_box?: number[][][];
  polygon?: number[][][];
  raster?: object;
  embedding?: number[];
}

export type Metric = {
  type: string;
  parameters?: Partial<Record<string, any>>;
  value: number | any;
  label?: Label;
};

export type Evaluation = {
  id: number;
  model_name: string;
  datum_filter: { dataset_names: string[]; object: any };
  parameters: { task_type: TaskType; object: any };
  status: 'pending' | 'running' | 'done' | 'failed' | 'deleting';
  metrics: Metric[];
  confusion_matrices: any[];
  created_at: Date;
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
    var datasets: Dataset[] = response.data;
    for (let index = 0, length = datasets.length; index < length; ++index) {
      datasets[index].metadata = decodeMetadata(datasets[index].metadata);
    }
    return datasets;
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
    response.data.metadata = decodeMetadata(response.data.metadata);
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
    metadata = encodeMetadata(metadata)
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
    var models: Model[] = response.data;
    for (let index = 0, length = models.length; index < length; ++index) {
      models[index].metadata = decodeMetadata(models[index].metadata);
    }
    return models;
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
    response.data.metadata = decodeMetadata(response.data.metadata)
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
    metadata = encodeMetadata(metadata)
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
   * Takes data from the backend response and converts it to an Evaluation object
   * by converting the datetime string to a `Date` object and replacing -1 metric values with
   * `null`.
   */
  private unmarshalEvaluation(evaluation: any): Evaluation {
    const updatedMetrics = evaluation.metrics.map((metric: Metric) => ({
      ...metric,
      value: metric.value === -1 ? null : metric.value
    }));
    return {
      ...evaluation,
      metrics: updatedMetrics,
      created_at: new Date(evaluation.created_at)
    };
  }

  /**
   * Creates a new evaluation or gets an existing one if an evaluation with the
   * same parameters already exists.
   *
   * @param model name of the model
   * @param dataset name of the dataset
   * @param taskType type of task
   * @param [iouThresholdsToCompute] list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP)
   * @param [iouThresholdsToReturn] list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`
   * @param [labelMap] mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models
   * @param [recallScoreThreshold] confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall
   * @param [computePrCurves] boolean which determines whether we calculate precision-recall curves or not
   * @param [prCurveIouThreshold] the IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5. Does nothing when compute_pr_curves is set to False or None
   *
   * @returns {Promise<Evaluation>}
   */
  public async createOrGetEvaluation(
    model: string,
    dataset: string,
    taskType: TaskType,
    iouThresholdsToCompute?: number[],
    iouThresholdsToReturn?: number[],
    labelMap?: number[][][],
    recallScoreThreshold?: number,
    computePrCurves?: boolean,
    prCurveIouThreshold?: number
  ): Promise<Evaluation> {
    const response = await this.client.post('/evaluations', {
      model_names: [model],
      datum_filter: { dataset_names: [dataset] },
      parameters: {
        task_type: taskType,
        iou_thresholds_to_compute:iouThresholdsToCompute,
        iou_thresholds_to_return: iouThresholdsToReturn,
        label_map: labelMap,
        recall_score_threshold: recallScoreThreshold,
        compute_pr_curves: computePrCurves,
        pr_curve_iou_threshold: prCurveIouThreshold
      }
    });
    return this.unmarshalEvaluation(response.data[0]);
  }

  /**
   * Creates new evaluations given a list of models, or gets existing ones if evaluations with the
   * same parameters already exists.
   *
   * @param models names of the models
   * @param dataset name of the dataset
   * @param taskType type of task
   * @param [iouThresholdsToCompute] list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP)
   * @param [iouThresholdsToReturn] list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`
   * @param [labelMap] mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models
   * @param [recallScoreThreshold] confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall
   * @param [computePrCurves] boolean which determines whether we calculate precision-recall curves or not
   * @param [prCurveIouThreshold] the IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5. Does nothing when compute_pr_curves is set to False or None
   *
   * @returns {Promise<Evaluation[]>}
   */
  public async bulkCreateOrGetEvaluations(
    models: string[],
    dataset: string,
    taskType: TaskType,
    iouThresholdsToCompute?: number[],
    iouThresholdsToReturn?: number[],
    labelMap?: any[][][],
    recallScoreThreshold?: number,
    computePrCurves?: boolean,
    prCurveIouThreshold?: number
  ): Promise<Evaluation[]> {
    const response = await this.client.post('/evaluations', {
      model_names: models,
      datum_filter: { dataset_names: [dataset] },
      parameters: {
        task_type: taskType,
        iou_thresholds_to_compute:iouThresholdsToCompute,
        iou_thresholds_to_return: iouThresholdsToReturn,
        label_map: labelMap,
        recall_score_threshold: recallScoreThreshold,
        compute_pr_curves: computePrCurves,
        pr_curve_iou_threshold: prCurveIouThreshold
       }
    });
    return response.data.map(this.unmarshalEvaluation);
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
    return response.data.map(this.unmarshalEvaluation);
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
   * @param datum valor datum
   * @param annotations valor annotations
   *
   * @returns {Promise<void>}
   */
  public async addGroundTruth(
    datasetName: string,
    datum: Datum,
    annotations: Annotation[]
  ): Promise<void> {
    datum.metadata = encodeMetadata(datum.metadata)
    for (let index = 0, length = annotations.length; index < length; ++index) {
      annotations[index].metadata = encodeMetadata(annotations[index].metadata);
    }
    return this.client.post('/groundtruths', [
      {
        dataset_name: datasetName,
        datum: datum,
        annotations: annotations
      }
    ]);
  }

  /**
   * Adds predictions from a model
   *
   * @param datasetName name of the dataset
   * @param modelName name of the model
   * @param datum valor datum
   * @param annotations valor annotations
   *
   * @returns {Promise<void>}
   */
  public async addPredictions(
    datasetName: string,
    modelName: string,
    datum: Datum,
    annotations: Annotation[]
  ): Promise<void> {
    datum.metadata = encodeMetadata(datum.metadata)
    for (let index = 0, length = annotations.length; index < length; ++index) {
      annotations[index].metadata = encodeMetadata(annotations[index].metadata);
    }
    return this.client.post('/predictions', [
      {
        dataset_name: datasetName,
        model_name: modelName,
        datum: datum,
        annotations: annotations
      }
    ]);
  }
}
