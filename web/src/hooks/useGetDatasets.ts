import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../common/api';

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: () => axios.get(`${BASE_URL}/datasets`)
  });
}
