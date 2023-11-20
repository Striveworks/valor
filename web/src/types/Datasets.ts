import { Datum } from './Datum';

export type Dataset = {
  name: string;
  from_video: boolean;
  href: string;
  description: string;
  type: Datum;
  finalized: boolean;
};
