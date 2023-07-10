import { Page, Typography } from '@striveworks/minerva';
import { Loading } from '../components/Loading';
import { useGetModels } from '../hooks/useGetModels';

export function Models() {
  const { isLoading, isError } = useGetModels();

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <>An error has occured</>;
  }

  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Models</Typography.PageTitle>
      </Page.Header>
      <Page.Content></Page.Content>
    </Page.Main>
  );
}
