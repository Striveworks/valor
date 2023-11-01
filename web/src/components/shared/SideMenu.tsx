import { useAuth0 } from '@auth0/auth0-react';
import { SideBar } from '@striveworks/minerva';
import { Link, useLocation } from 'react-router-dom';

export function SideMenu() {
  const { user } = useAuth0();
  const location = useLocation();
  const basePath = location.pathname.split('/')[1];

  return (
    <SideBar>
      <SideBar.Logo>Velour</SideBar.Logo>
      <SideBar.Nav>
        <Link to={'/'} style={{ textDecoration: 'none' }}>
          <SideBar.Tab
            iconName='home'
            title='Home'
            selected={basePath === '' ? true : false}
          >
            Home
          </SideBar.Tab>
        </Link>
        <Link to={'/models'} style={{ textDecoration: 'none' }}>
          <SideBar.Tab
            iconName='model'
            title='Models'
            selected={basePath === 'models' ? true : false}
          >
            Model
          </SideBar.Tab>
        </Link>
        <Link to={'/datasets'} style={{ textDecoration: 'none' }}>
          <SideBar.Tab
            iconName='dataset'
            title='Datasets'
            selected={basePath === 'datasets' ? true : false}
          >
            Datasets
          </SideBar.Tab>
        </Link>
        <Link to={'/evaluations'} style={{ textDecoration: 'none' }}>
          <SideBar.Tab
            iconName='vectorPolygon'
            title='Evaluations'
            selected={basePath === 'evaluations' ? true : false}
          >
            Evaluations
          </SideBar.Tab>
        </Link>
        {user !== undefined && (
          <Link to={'/profile'} style={{ textDecoration: 'none' }}>
            <SideBar.Tab
              iconName='user'
              title='Profile'
              selected={basePath === 'profile' ? true : false}
            >
              Profile
            </SideBar.Tab>
          </Link>
        )}
        <SideBar.Tab
          iconName='docs'
          title='Documentation'
          target={'_blank'}
          href='https://striveworks.github.io/velour/'
        >
          Documentation
        </SideBar.Tab>
      </SideBar.Nav>
    </SideBar>
  );
}
