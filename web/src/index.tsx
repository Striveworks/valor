import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { Auth0ProviderWithNavigate, usingAuth } from './auth';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <BrowserRouter>
      {usingAuth() ? (
        <Auth0ProviderWithNavigate>
          <App />
        </Auth0ProviderWithNavigate>
      ) : (
        <>
          <App />
        </>
      )}
    </BrowserRouter>
  </React.StrictMode>
);
