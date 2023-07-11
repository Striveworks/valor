import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../../common/api';
import { ModelMetric } from '../../types/Models';

export function useGetModelDetails(modelName: string) {
  return useQuery<ModelMetric[], Error>({
    queryKey: ['models', modelName],
    queryFn: async () => {
      const response = await axios.get(
        `${BASE_URL}/models/${modelName}/evaluation-settings`
      );
      return response.data;
    }
  });
}
