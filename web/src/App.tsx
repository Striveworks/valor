import { useAuth0 } from '@auth0/auth0-react';
import { ChariotLayoutTemplate } from '@striveworks/minerva';
import '@striveworks/minerva/style.css';
import { Route, Routes } from 'react-router-dom';
import './App.css';
import { usingAuth } from './Auth';
import { SideMenu } from './components/shared/SideMenu';
import { DatasetDetailsPage } from './dataset-details-page';
import { MetricsPage } from './metrics-page';
import { Datasets } from './pages/Datasets/Datasets';
import { Home } from './pages/Home';
import { Login } from './pages/Login';
import { ModelDetails } from './pages/Models/ModelDetails';
import { Models } from './pages/Models/Models';
import { Profile } from './pages/Profile';
import { Evaluations } from './pages/Evaluations/Evaluations';

function App() {
  const { isAuthenticated } = useAuth0();

  if (isAuthenticated || !usingAuth()) {
    return (
      <ChariotLayoutTemplate SideBar={<SideMenu />}>
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/models'>
            <Route index element={<Models />} />
            <Route path=':modelName' element={<ModelDetails />} />
            <Route
              path=':name/evaluation-settings/:evalSettingsId'
              element={<MetricsPage />}
            />
          </Route>
          <Route path='/datasets'>
            <Route index element={<Datasets />} />
            <Route path=':name' element={<DatasetDetailsPage />} />
          </Route>
          <Route path='/evaluations'>
            <Route index element={<Evaluations />} />
          </Route>
          <Route path='/profile' element={<Profile />} />
        </Routes>
      </ChariotLayoutTemplate>
    );
  }

  return <Login />;
}

export default App;
