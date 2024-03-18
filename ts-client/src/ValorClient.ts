import axios, { AxiosInstance } from 'axios';

type Dataset = {
  id: number;
  name: string;
  metadata: object;
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

  public async deleteDataset(name: string): Promise<void> {
    await this.client.delete(`/datasets/${name}`);
  }
}
