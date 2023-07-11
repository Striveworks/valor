import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../common/api';
import { Model } from '../types/Models';

export function useGetModels() {
  return useQuery<Model[], Error>({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await axios.get(`${BASE_URL}/models`);
      return response.data;
    }
  });
}
