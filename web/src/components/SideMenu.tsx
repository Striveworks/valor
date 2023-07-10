import { useAuth0 } from '@auth0/auth0-react';
import { Sidebar } from '@striveworks/minerva';
import { Link, useLocation } from 'react-router-dom';

export function SideMenu() {
  const { user } = useAuth0();
  const location = useLocation();
  const basePath = location.pathname.split('/')[1];

  return (
    <Sidebar>
      <Sidebar.Logo>Velour</Sidebar.Logo>
      <Sidebar.Nav>
        <Link to={'/'}>
          <Sidebar.Tab
            iconName='home'
            tooltip='Home'
            isSelected={basePath === '' ? true : false}
            href={''}
          >
            Home
          </Sidebar.Tab>
        </Link>
        <Link to={'/models'}>
          <Sidebar.Tab
            iconName='model'
            tooltip='Models'
            isSelected={basePath === 'models' ? true : false}
          >
            Model
          </Sidebar.Tab>
        </Link>
        <Link to={'/datasets'}>
          <Sidebar.Tab
            iconName='dataset'
            tooltip='Datasets'
            isSelected={basePath === 'datasets' ? true : false}
          >
            Datasets
          </Sidebar.Tab>
        </Link>
        {user !== undefined && (
          <Link to={'/profile'}>
            <Sidebar.Tab
              iconName='user'
              tooltip='Profile'
              isSelected={basePath === 'profile' ? true : false}
            >
              Profile
            </Sidebar.Tab>
          </Link>
        )}
        <Sidebar.Tab
          iconName='docs'
          tooltip='Documentation'
          target={'_blank'}
          href='https://striveworks.github.io/velour/'
        >
          Documentation
        </Sidebar.Tab>
      </Sidebar.Nav>
    </Sidebar>
  );
}
