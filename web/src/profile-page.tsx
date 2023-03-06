import { useEffect, useState } from "react";
import { useAuth0 } from "@auth0/auth0-react";

export const ProfilePage = () => {
  const [accessToken, setAccessToken] = useState("");

  console.log("inside profile");
  const { user, getAccessTokenSilently } = useAuth0();
  console.log("user:");
  console.log(user);

  useEffect(() => {
    const getToken = async () => {
      const token = await getAccessTokenSilently();
      console.log(`token inside getToken: ${token}`);
      setAccessToken(token);
    };
    getToken();
  });

  console.log(`accessToken: ${accessToken}`);
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

// {fields.map((f) => (
//   <p>{f}</p>
// ))}

// <tr>
//               <th>f</th>
//               <td>
//                 <p>asd</p>
//               </td>
//             </tr>
