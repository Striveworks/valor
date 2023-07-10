import { useAuth0 } from '@auth0/auth0-react';
import { ChariotLayoutTemplate } from '@striveworks/minerva';
import '@striveworks/minerva/style.css';
import { Route, Routes } from 'react-router-dom';
import './App.css';
import { usingAuth } from './auth';
import { SideMenu } from './components/shared/SideMenu';
import { DatasetDetailsPage } from './dataset-details-page';
import { MetricsPage } from './metrics-page';
import { ModelDetailsPage } from './model-details-page';
import { Datasets } from './pages/Datasets';
import { Home } from './pages/Home';
import { Login } from './pages/Login';
import { Models } from './pages/Models';
import { Profile } from './pages/Profile';

function App() {
  const { isAuthenticated } = useAuth0();

  if (isAuthenticated || !usingAuth()) {
    return (
      <ChariotLayoutTemplate sidebar={<SideMenu />}>
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/models' element={<Models />} />
          <Route path='/models/:name' element={<ModelDetailsPage />} />
          <Route path='/datasets' element={<Datasets />} />
          <Route path='/datasets/:name' element={<DatasetDetailsPage />} />
          <Route
            path='/models/:name/evaluation-settings/:evalSettingsId'
            element={<MetricsPage />}
          />
          <Route path='/profile' element={<Profile />} />
        </Routes>
      </ChariotLayoutTemplate>
    );
  }

  return <Login />;
}

export default App;
