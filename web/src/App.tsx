import "./App.css";
import { Route, Routes } from "react-router-dom";
import { CallbackPage } from "./callback-page";
import { ProfilePage } from "./profile-page";
import { LoginButton } from "./login-button";
import { ModelsPage } from "./models-page";
import { usingAuth } from "./auth";

function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <div className="App">
            <header className="App-header">
              <h1>velour</h1>
              {usingAuth() ? <LoginButton /> : <></>}
            </header>
          </div>
        }
      />
      <Route path="/callback" element={<CallbackPage />} />
      <Route path="/profile" element={<ProfilePage />} />
      <Route path="/models" element={<ModelsPage />} />
    </Routes>
  );
}

export default App;
