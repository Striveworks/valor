import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../common/api';

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => axios.get(`${BASE_URL}/models`)
  });
}
