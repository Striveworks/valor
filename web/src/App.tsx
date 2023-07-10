import { useAuth0 } from '@auth0/auth0-react';
import { ChariotLayoutTemplate } from '@striveworks/minerva';
import { Route, Routes } from 'react-router-dom';
import './App.css';
import { usingAuth } from './auth';
import { ListingComponent } from './components/listing-component';
import { SideMenu } from './components/SideMenu';
import { DatasetDetailsPage } from './dataset-details-page';
import { MetricsPage } from './metrics-page';
import { ModelDetailsPage } from './model-details-page';
import { Home } from './pages/Home';
import { Login } from './pages/Login';
import '@striveworks/minerva/style.css';
import { Profile } from './pages/Profile';

function App() {
  const { isAuthenticated } = useAuth0();

  if (isAuthenticated || !usingAuth()) {
    return (
      <ChariotLayoutTemplate sidebar={<SideMenu />}>
        <Routes>
          <Route path='/' element={<Home />} />
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
          <Route path='/profile' element={<Profile />} />
          <Route
            path='/models/:name/evaluation-settings/:evalSettingsId'
            element={<MetricsPage />}
          />
        </Routes>
      </ChariotLayoutTemplate>
    );
  }

  return <Login />;
}

export default App;
