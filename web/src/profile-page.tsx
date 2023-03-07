import { useEffect, useState } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { LogoutButton } from "./logout-button";
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/default-highlight";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";

export const ProfilePage = () => {
  const [accessToken, setAccessToken] = useState("");
  const { user, getAccessTokenSilently } = useAuth0();

  useEffect(() => {
    const getToken = async () => {
      const token = await getAccessTokenSilently();
      setAccessToken(token);
    };
    getToken();
  });

  const fields = ["name", "email"];
  return (
    <div>
      <h2>User Information</h2>
      <table>
        <tbody>
          {fields.map((f) => (
            <tr key={f}>
              <th>{f}:</th>
              <td>{user ? user[f] : null}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2>
        The python snippet below establishes an authenticated connection to the
        velour instance.
      </h2>

      <div style={{ width: "75%" }}>
        <SyntaxHighlighter language="python" style={atomOneDark}>
          {`from velour.client import Client\n\nclient = Client("${process.env.REACT_APP_AUTH0_AUDIENCE}", access_token="${accessToken}")`}
        </SyntaxHighlighter>
      </div>
      <LogoutButton />
    </div>
  );
};
