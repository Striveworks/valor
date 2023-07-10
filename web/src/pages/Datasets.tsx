import { Page, Typography } from '@striveworks/minerva';
import { Loading } from '../components/Loading';
import { useGetDatasets } from '../hooks/useGetDatasets';

export function Datasets() {
  const { isLoading, isError, data } = useGetDatasets();

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <>An error has occured</>;
  }

  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Datasets</Typography.PageTitle>
      </Page.Header>
      <Page.Content></Page.Content>
    </Page.Main>
  );
}
