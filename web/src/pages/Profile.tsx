import { useAuth0 } from '@auth0/auth0-react';
import { Box, Button, Page, Typography } from '@striveworks/minerva';
import { useState } from 'react';
import CopyToClipboard from 'react-copy-to-clipboard';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { LogoutButton } from '../components/shared/LogoutButton';

export function Profile() {
  const { user } = useAuth0();
  const [snippetCopied, setSnippetCopied] = useState(false);

  const codeSnippet = `from velour.client import Client\n\nclient = Client("${
    import.meta.env.VITE_BACKEND_URL
  }", access_token="${sessionStorage.getItem('token')}")`;

  // const fields = ['name', 'email'];
  return (
    <Page.Main>
      <Page.Header>
        <Typography.PageTitle>Profile</Typography.PageTitle>
      </Page.Header>
      <Page.Content>
        <Box direction='column' gap='4rem'>
          <Box direction='column' gap='1rem'>
            <Typography.SectionTitle>User Information</Typography.SectionTitle>
            <p>Name: {user?.name}</p>
            <p>Email: {user?.email}</p>
          </Box>

          <Box direction='column' gap='1rem'>
            <Typography.SectionTitle>Connecting Velour</Typography.SectionTitle>
            <Typography.SectionTitle>
              The python snippet below establishes an authenticated connection to the
              velour instance.
            </Typography.SectionTitle>
            <SyntaxHighlighter language='python' style={atomOneDark}>
              {codeSnippet}
            </SyntaxHighlighter>
            <CopyToClipboard text={codeSnippet} onCopy={() => setSnippetCopied(true)}>
              <Button iconName='clipboard'>Copy</Button>
            </CopyToClipboard>
            {snippetCopied && <span style={{ fontWeight: 'bolder' }}> copied! </span>}
          </Box>

          <Box direction='column' gap='1rem'>
            <Typography.SectionTitle>Log Out</Typography.SectionTitle>
            <LogoutButton />
          </Box>
        </Box>
      </Page.Content>
    </Page.Main>
  );
}
