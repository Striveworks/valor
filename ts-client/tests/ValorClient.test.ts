// integration tests against a live valor instance running on http://localhost:8000

import { ValorClient } from '../src/ValorClient';

const baseURL = 'http://localhost:8000';
const client = new ValorClient(baseURL);

beforeEach(async () => {
  // make sure there are no datasets or models in the backend
  const datasets = await client.getAllDatasets();
  const models = await client.getAllModels();
  if (datasets.length > 0 || models.length > 0) {
    throw new Error('Valor backend is not empty');
  }
});

afterEach(async () => {
  // delete any datasets or models in the backend
  // sleep
  await new Promise((resolve) => setTimeout(resolve, 500));
  const datasets = await client.getAllDatasets();
  datasets.forEach(async (dataset) => {
    await client.deleteDataset(dataset.name);
  });
  const models = await client.getAllModels();
  models.forEach(async (model) => {
    await client.deleteModel(model.name);
  });

  // sleep for a bit to allow the backend to delete the datasets and models
  await new Promise((resolve) => setTimeout(resolve, 100));
});

test('dataset methods', async () => {
  await client.createDataset('test-dataset1', { k1: 'v1', k2: 'v2' });
  await client.createDataset('test-dataset2', { k1: 'v2', k3: 'v3' });

  // check we can get all datasets
  const allDatasets = await client.getAllDatasets();
  expect(Array.isArray(allDatasets)).toBe(true);
  expect(allDatasets.length).toBe(2);
  const datasetNames = allDatasets.map((dataset) => dataset.name);
  expect(datasetNames).toEqual(
    expect.arrayContaining(['test-dataset1', 'test-dataset2'])
  );

  // check we can get a dataset by metadata
  const datasetsByMetadata1 = await client.getDatasetsByMetadata({ k1: 'v1' });
  expect(datasetsByMetadata1.length).toBe(1);
  expect(datasetsByMetadata1[0].name).toBe('test-dataset1');

  const datasetsByMetadata2 = await client.getDatasetsByMetadata({ k1: 'v3' });
  expect(datasetsByMetadata2.length).toBe(0);
});

test('model methods', async () => {
  await client.createModel('test-model1', { k1: 'v1', k2: 'v2' });
  await client.createModel('test-model2', { k1: 'v2', k3: 'v3' });

  // check we can get all models
  const allModels = await client.getAllModels();
  expect(Array.isArray(allModels)).toBe(true);
  expect(allModels.length).toBe(2);
  const modelNames = allModels.map((model) => model.name);
  expect(modelNames).toEqual(expect.arrayContaining(['test-model1', 'test-model2']));

  // check we can get a model by metadata
  const modelsByMetadata1 = await client.getModelsByMetadata({ k1: 'v1' });
  expect(modelsByMetadata1.length).toBe(1);
  expect(modelsByMetadata1[0].name).toBe('test-model1');

  const modelsByMetadata2 = await client.getModelsByMetadata({ k1: 'v3' });
  expect(modelsByMetadata2.length).toBe(0);
});

test('evaluation methods', async () => {
  await client.createDataset('test-dataset1', {});
  await client.createModel('test-model1', {});

  await client.addGroundTruth('test-dataset1', 'uid1', [
    { task_type: 'classification', labels: [{ key: 'label-key', value: 'label-value' }] }
  ]);
  await client.finalizeDataset('test-dataset1');
  await client.addPredictions('test-model1', 'test-dataset1', 'uid1', [
    {
      task_type: 'classification',
      labels: [{ key: 'label-key', value: 'label-value', score: 1.0 }]
    }
  ]);

  let evaluation = await client.createOrGetEvaluation(
    'test-model1',
    'test-dataset1',
    'classification'
  );
  expect(['running', 'pending', 'done']).toContain(evaluation.status);

  // poll until the evaluation is done
  while (evaluation.status !== 'done') {
    await new Promise((resolve) => setTimeout(resolve, 100));
    evaluation = await client.getEvaluationById(evaluation.id);
  }
  expect(evaluation.metrics.length).toBeGreaterThan(0);

  // check we can get my model name
  const modelEvaluations = await client.getEvaluationsByModelName('test-model1');
  expect(modelEvaluations.length).toBe(1);

  const noModelEvaluations = await client.getEvaluationsByModelName('no-such-model');
  expect(noModelEvaluations.length).toBe(0);

  // check we can get my dataset name
  const datasetEvaluations = await client.getEvaluationsByDatasetName('test-dataset1');
  expect(datasetEvaluations.length).toBe(1);

  const noDatasetEvaluations = await client.getEvaluationsByDatasetName(
    'no-such-dataset'
  );
  expect(noDatasetEvaluations.length).toBe(0);
});
