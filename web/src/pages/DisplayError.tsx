import { Page, Typography } from '@striveworks/minerva';

export function DisplayError({ error }: { error: Error }) {
  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Error</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <Typography.SectionTitle>
          Sorry, an unexpected error has occured.
        </Typography.SectionTitle>
        <p>{error.message}</p>
      </Page.Content>
    </Page.Main>
  );
}
