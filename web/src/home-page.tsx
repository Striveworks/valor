import { usingAuth } from "./auth";
import Typography from "@mui/material/Typography";
import { Wrapper } from "./components/wrapper";
import { LoginButton } from "./login-button";
import { useAuth0 } from "@auth0/auth0-react";

export const HomePage = () => {
  const { isAuthenticated } = useAuth0();

  const content = (
    <div className="App">
      <header className="App-header">
        <Typography variant="h1">velour</Typography>
        <br />
        {usingAuth() && !isAuthenticated ? <LoginButton /> : <></>}
      </header>
    </div>
  );

  if (!usingAuth() || isAuthenticated) {
    return <Wrapper>{content}</Wrapper>;
  }
  return content;
};
