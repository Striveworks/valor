import "./App.css";
import { Route, Routes } from "react-router-dom";
import { CallbackPage } from "./callback-page";
import { ProfilePage } from "./profile-page";
import { LoginButton } from "./login-button";

function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <div className="App">
            <header className="App-header">
              <h1>velour</h1>
              <LoginButton />
            </header>
          </div>
        }
      />
      <Route path="/callback" element={<CallbackPage />} />
      <Route path="/profile" element={<ProfilePage />} />
    </Routes>
  );
}

export default App;
