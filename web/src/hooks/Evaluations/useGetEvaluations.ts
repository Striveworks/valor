import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { BASE_URL } from '../../common/api';
import { Evaluation } from '../../types/Evaluations';

export function useGetEvaluations() {
    return useQuery<Evaluation[], Error>({
        queryKey: ['evaluations'],
        queryFn: async () => {
            const response = await axios.get(`${BASE_URL}/evaluations`);
            return response.data;
        }
    });
}
