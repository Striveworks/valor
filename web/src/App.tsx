import "./App.css";
import { Route, Routes } from "react-router-dom";
import { CallbackPage } from "./callback-page";
import { ProfilePage } from "./profile-page";
import { LoginButton } from "./login-button";
import { ModelsPage } from "./models-page";
import { ModelDetailsPage } from "./model-details-page";
import { usingAuth } from "./auth";
import { MetricsPage } from "./metrics-page";

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
      <Route path="/models/:name" element={<ModelDetailsPage />} />
      <Route
        path="/models/:name/evaluation-settings/:evalSettingsId"
        element={<MetricsPage />}
      />
    </Routes>
  );
}

export default App;
