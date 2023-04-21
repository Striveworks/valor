import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { Auth0ProviderWithNavigate, usingAuth } from "./auth";
import reportWebVitals from "./reportWebVitals";
import { ThemeProvider } from "@mui/material/styles";
import { createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { StyledEngineProvider } from "@mui/material/styles";
// import AnekTamil from "@fontsource/anek-tamil";

const theme = createTheme({
  typography: {
    fontFamily: "Anek Tamil",
    allVariants: {
      color: "white",
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundImage: `linear-gradient(to right top, #4c4355, #3d323e, #2c2329, #1a1517, #000000)`,
        },
        html: {
          height: `100%`,
        },
        table: {
          width: `50%`,
          whiteSpace: `nowrap`,
          tableLayout: `fixed`,
          padding: `15px`,
        },
        td: {
          fontWeight: `bold`,
        },
        th: {
          fontWeight: `bolder`,
          width: `25%`,
        },
        "td, th": {
          textAlign: `left`,
          verticalAlign: `top`,
          padding: `0 0px`,
        },
      },
    },
  },
});

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);
root.render(
  <React.StrictMode>
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          {usingAuth() ? (
            <Auth0ProviderWithNavigate>
              <App />
            </Auth0ProviderWithNavigate>
          ) : (
            <App />
          )}
        </BrowserRouter>
      </ThemeProvider>
    </StyledEngineProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
