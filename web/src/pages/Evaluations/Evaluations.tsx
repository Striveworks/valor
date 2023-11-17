import { Page, TableList, Typography } from '@striveworks/minerva';
import { Loading } from '../../components/shared/Loading';
import { SummaryBar } from '../../components/shared/SummaryBar';
import { useGetEvaluations } from '../../hooks/Evaluations/useGetEvaluations';
import { Stat } from '../../types/TableList';
import { DisplayError } from '../DisplayError';

export function Evaluations() {
    const { isLoading, isError, data, error } = useGetEvaluations();

    if (isLoading) {
        return <Loading />;
    }

    if (isError) {
        return <DisplayError error={error} />;
    }

    const stats: Stat[] = [{ name: data.length, icon: 'collection' }];

    return (
        <Page.Main>
            <Page.Header>
                <Typography textStyle='titlePageLarge'>Evaluations</Typography>
            </Page.Header>
            <Page.Content>
                <TableList summaryBar={<SummaryBar stats={stats} />}>
                    {data.length ? (
                        data.map((evaluation) => {
                            return (
                                <TableList.Row key={evaluation.dataset}></TableList.Row>
                            );
                        })
                    ) : (
                        <TableList.Row>No Items Found</TableList.Row>
                    )}
                </TableList>
            </Page.Content>
        </Page.Main>
    );
}
