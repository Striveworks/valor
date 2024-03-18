import { ValorClient } from '../src/ValorClient';

const baseURL = 'http://localhost:8000';
const client = new ValorClient(baseURL);

beforeEach(async () => {
  // make sure there are no datasets or models in the backend
  const datasets = await client.getDatasets({});
  const models = await client.getModels({});
  if (datasets.length > 0 || models.length > 0) {
    throw new Error('Valor backend is not empty');
  }
});

afterEach(async () => {
  // delete any datasets or models in the backend
  const datasets = await client.getDatasets({});
  datasets.forEach(async (dataset) => {
    await client.deleteDataset(dataset.name);
  });
  const models = await client.getModels({});
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
  const allDatasets = await client.getDatasets({});
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
  const allModels = await client.getModels({});
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
