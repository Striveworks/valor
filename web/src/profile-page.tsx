// TODO: Delete Page
export {};
// import { useAuth0 } from '@auth0/auth0-react';
// import Button from '@mui/material/Button';
// import { useState } from 'react';
// import { CopyToClipboard } from 'react-copy-to-clipboard';
// import SyntaxHighlighter from 'react-syntax-highlighter';
// import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
// import { Wrapper } from './components/wrapper';
// import { LogoutButton } from './logout-button';

// export const ProfilePage = () => {
//   const { user } = useAuth0();
//   const [snippetCopied, setSnippetCopied] = useState(false);

//   const codeSnippet = `from velour.client import Client\n\nclient = Client("${
//     import.meta.env.VITE_BACKEND_URL
//   }", access_token="${sessionStorage.getItem('token')}")`;

//   const fields = ['name', 'email'];
//   return (
//     <Wrapper>
//       <h2>User Information</h2>
//       <table>
//         <tbody>
//           {fields.map((f) => (
//             <tr key={f}>
//               <th>{f}:</th>
//               <td>{user ? user[f] : null}</td>
//             </tr>
//           ))}
//         </tbody>
//       </table>
//       <h2>
//         The python snippet below establishes an authenticated connection to the velour
//         instance.
//       </h2>

//       <div style={{ width: '75%' }}>
//         <SyntaxHighlighter language='python' style={atomOneDark}>
//           {codeSnippet}
//         </SyntaxHighlighter>
//       </div>
//       <div>
//         <div>
//           <CopyToClipboard text={codeSnippet} onCopy={() => setSnippetCopied(true)}>
//             <Button variant='contained'>Copy code to clipboard</Button>
//           </CopyToClipboard>
//           {snippetCopied ? (
//             <span style={{ fontWeight: 'bolder' }}> copied! </span>
//           ) : (
//             <></>
//           )}
//         </div>
//         <br />
//         <LogoutButton />
//       </div>
//     </Wrapper>
//   );
// };
