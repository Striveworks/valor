import { ValorClient } from '../src/ValorClient';

const baseURL = 'http://localhost:8000';
const client = new ValorClient(baseURL);

beforeEach(async () => {
  // make sure there are no datasets or models in the backend
  const datasets = await client.getDatasets({});
  if (datasets.length > 0) {
    throw new Error('Valor backend is not empty');
  }
});

afterEach(async () => {
  // delete any datasets or models in the backend
  const datasets = await client.getDatasets({});
  datasets.forEach(async (dataset) => {
    await client.deleteDataset(dataset.name);
  });
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
