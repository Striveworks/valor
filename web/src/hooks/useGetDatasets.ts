import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../common/api';
import { Dataset } from '../types/Datasets';

export function useGetDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: async (): Promise<Dataset[]> => {
      const response = await axios.get(`${BASE_URL}/datasets`);
      return response.data;
    }
  });
}
