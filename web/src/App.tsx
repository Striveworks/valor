import { Route, Routes } from 'react-router-dom';
import './App.css';
import { CallbackPage } from './callback-page';
import { ListingComponent } from './components/listing-component';
import { DatasetDetailsPage } from './dataset-details-page';
import { HomePage } from './home-page';
import { MetricsPage } from './metrics-page';
import { ModelDetailsPage } from './model-details-page';
import { ProfilePage } from './profile-page';

function App() {
  return (
    <Routes>
      <Route path='/' element={<HomePage />} />
      <Route path='/callback' element={<CallbackPage />} />
      <Route path='/profile' element={<ProfilePage />} />
      <Route
        path='/models'
        element={<ListingComponent name='models' pageTitle='Models' />}
      />
      <Route
        path='/datasets'
        element={<ListingComponent name='datasets' pageTitle='Datasets' />}
      />
      <Route path='/models/:name' element={<ModelDetailsPage />} />
      <Route path='/datasets/:name' element={<DatasetDetailsPage />} />
      <Route
        path='/models/:name/evaluation-settings/:evalSettingsId'
        element={<MetricsPage />}
      />
    </Routes>
  );
}

export default App;
