import { useEffect, useState } from "react";
import { useAuth0 } from "@auth0/auth0-react";

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
      <table>
        {fields.map((f) => (
          <tr>
            <th>{f}:</th>
            <td>{user ? user[f] : null}</td>
          </tr>
        ))}
        <tr>
          <th>Access token:</th>
          <td>{accessToken}</td>
        </tr>
      </table>
    </div>
  );
};
