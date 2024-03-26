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
  const datasets = await client.getAllDatasets();
  await Promise.all(
    datasets.map(async (dataset) => {
      await client.deleteDataset(dataset.name);
    })
  );

  const models = await client.getAllModels();
  await Promise.all(
    models.map(async (model) => {
      await client.deleteModel(model.name);
    })
  );

  // wait for all models and datasets to be deleted
  while (
    (await client.getAllModels()).length > 0 &&
    (await client.getAllDatasets()).length > 0
  ) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
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

/**
 * Helper method that creates two datasets with groundtruth and two models with predictions
 * on each dataset
 */
const createDatasetsAndModels = async () => {
  const datasetNames = ['test-dataset1', 'test-dataset2'];
  const modelNames = ['test-model1', 'test-model2'];

  // create datasets and add groundtruths
  for (const datasetName of datasetNames) {
    await client.createDataset(datasetName, {});
    await client.addGroundTruth(datasetName, 'uid1', [
      {
        task_type: 'classification',
        labels: [{ key: 'label-key', value: 'label-value' }]
      }
    ]);
    await client.finalizeDataset(datasetName);
  }

  // create models and add predictions
  await Promise.all(
    modelNames.map(async (modelName) => {
      await client.createModel(modelName, {});

      await Promise.all(
        datasetNames.map(async (datasetName) => {
          await client.addPredictions(modelName, datasetName, 'uid1', [
            {
              task_type: 'classification',
              labels: [{ key: 'label-key', value: 'label-value', score: 1.0 }]
            }
          ]);
        })
      );
    })
  );

  return { datasetNames, modelNames };
};

test('evaluation methods', async () => {
  const { datasetNames, modelNames } = await createDatasetsAndModels();

  // evals a model against a dataset and polls the status
  const evalAndWaitForCompletion = async (modelName: string, datasetName: string) => {
    let evaluation = await client.createOrGetEvaluation(
      modelName,
      datasetName,
      'classification'
    );
    expect(['running', 'pending', 'done']).toContain(evaluation.status);

    while (evaluation.status !== 'done') {
      await new Promise((resolve) => setTimeout(resolve, 100));
      evaluation = await client.getEvaluationById(evaluation.id);
    }
    expect(evaluation.metrics.length).toBeGreaterThan(0);
    expect(evaluation.datum_filter.dataset_names).toStrictEqual([datasetName]);
  };
  // evaluate against all models and datasets
  await Promise.all(
    modelNames.map(async (modelName) => {
      await Promise.all(
        datasetNames.map(async (datasetName) => {
          await evalAndWaitForCompletion(modelName, datasetName);
        })
      );
    })
  );
  // check we can get evaluations by model names
  expect((await client.getEvaluationsByModelNames([modelNames[0]])).length).toBe(2);
  expect((await client.getEvaluationsByModelNames(modelNames)).length).toBe(4);
  expect((await client.getEvaluationsByModelNames(['no-such-model'])).length).toBe(0);
  // check we can get evaluations my dataset name
  expect((await client.getEvaluationsByDatasetNames([datasetNames[0]])).length).toBe(2);
  expect((await client.getEvaluationsByDatasetNames(datasetNames)).length).toBe(4);
  expect((await client.getEvaluationsByDatasetNames(['no-such-dataset'])).length).toBe(0);

  // test bulk create or get evaluations
});

test('bulk create or get evaluations', async () => {
  const { datasetNames, modelNames } = await createDatasetsAndModels();

  // bulk create evaluations for each dataset
  for (const datasetName of datasetNames) {
    await client.finalizeDataset(datasetName);

    let evaluations = await client.bulkCreateOrGetEvaluations(
      modelNames,
      datasetName,
      'classification'
    );
    expect(evaluations.length).toBe(2);
    // check all evaluations are pending

    while (evaluations.every((evaluation) => evaluation.status !== 'done')) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      evaluations = await client.getEvaluationsByIds(
        evaluations.map((evaluation) => evaluation.id)
      );
      expect(evaluations.length).toBe(2);
    }
  }
});
