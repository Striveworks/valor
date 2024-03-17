import axios, { AxiosInstance } from 'axios';

type Point = {
  x: number;
  y: number;
};

type BasicPolygon = {
  points: Point[];
};

type Polygon = {
  boundary: BasicPolygon;
  holes: BasicPolygon[];
};

type MultiPolygon = {
  polygons: Polygon[];
};

type BoundingBox = {
  polygon: BasicPolygon;
};

type Raster = {
  mask?: string;
  geometry?: BoundingBox | Polygon | MultiPolygon;
};

type Datum = {
  uid: string;
  dataset_name: string;
  metadata: any;
};

type Label = {
  key: string;
  value: string;
};

type Annotation = {
  task_type: string;
  labels: Label[];
  metadata?: any;
  bounding_box?: BoundingBox;
  polygon?: Polygon;
  raster?: Raster;
  embedding?: number[];
};

type GroundTruth = {
  datum: Datum;
  annotations: Annotation[];
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

  public async createGroundTruths(groundTruths: GroundTruth[]): Promise<void> {
    await this.client.post('/groundtruths', groundTruths);
  }
}
