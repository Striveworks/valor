import "./App.css";
import { Route, Routes } from "react-router-dom";
import { CallbackPage } from "./callback-page";
import { ProfilePage } from "./profile-page";
import { ModelDetailsPage } from "./model-details-page";

import { MetricsPage } from "./metrics-page";
import { HomePage } from "./home-page";
import { ListingComponent } from "./components/listing-component";
import { EntityDetailsComponent } from "./components/entity-details-component";
import { DatasetDetailsPage } from "./dataset-details-page";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/callback" element={<CallbackPage />} />
      <Route path="/profile" element={<ProfilePage />} />
      <Route
        path="/models"
        element={<ListingComponent name="models" pageTitle="Models" />}
      />
      <Route
        path="/datasets"
        element={<ListingComponent name="datasets" pageTitle="Datasets" />}
      />
      <Route path="/models/:name" element={<ModelDetailsPage />} />
      <Route path="/datasets/:name" element={<DatasetDetailsPage />} />
      <Route
        path="/models/:name/evaluation-settings/:evalSettingsId"
        element={<MetricsPage />}
      />
    </Routes>
  );
}

export default App;
