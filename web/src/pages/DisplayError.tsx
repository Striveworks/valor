import { Page, Typography } from '@striveworks/minerva';

export function DisplayError({ error }: { error: Error }) {
  return (
    <Page.Main>
      <Page.Header>
        <Typography textStyle='titlePageLarge'>Error</Typography>
      </Page.Header>
      <Page.Content
        xcss={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}
      >
        <Typography>
          Sorry, an unexpected error has occurred.
        </Typography>
        <p>{error.message}</p>
      </Page.Content>
    </Page.Main>
  );
}
