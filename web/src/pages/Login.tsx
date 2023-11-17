import { useAuth0 } from '@auth0/auth0-react';
import { Button, ChariotLayoutTemplate, Page, Typography } from '@striveworks/minerva';

export function Login() {
  const { loginWithRedirect } = useAuth0();

  const handleLogin = async () => {
    await loginWithRedirect({
      appState: {
        returnTo: '/'
      }
    });
  };

  return (
    <ChariotLayoutTemplate>
      <Page.Main
        xcss={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Page.Header>
          <Typography textStyle='titlePageLarge'>Velour</Typography>
        </Page.Header>
        <Page.Content>
          <Button onClick={handleLogin}>Login</Button>
        </Page.Content>
      </Page.Main>
    </ChariotLayoutTemplate>
  );
}
