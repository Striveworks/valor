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
import {
  Link as RouterLink,
  LinkProps as RouterLinkProps,
} from "react-router-dom";
import { LinkProps } from "@mui/material/Link";
import type {} from "@mui/x-data-grid/themeAugmentation";

const LinkBehavior = React.forwardRef<
  HTMLAnchorElement,
  Omit<RouterLinkProps, "to"> & { href: RouterLinkProps["to"] }
>((props, ref) => {
  const { href, ...other } = props;
  // Map href (MUI) -> to (react-router)
  return <RouterLink ref={ref} to={href} {...other} />;
});

export const StriveColors = {
  logoPurple: "#820782",
  logoOutline: "#25034c",
  strongPurple: "#7209b7",
  ctaGreen: "#36CFC9",
  lightPurple: "#AA6CFF",
  transitionBlue: "#3A0CA3",
  brightBlue: "#426BF1",
  transitionSeafoam: "#A6CEE3",
  altButton: "#F0F0F0",
  backgroundPrimary: "#2C3550",
  backgroundDark: "#232D49",
  placeholder: "#7E89A9",
  breadCrumb: "#FCE26B",
  error: "#E64444",
  success: "#1CBA46",
};

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: StriveColors.brightBlue,
      dark: StriveColors.backgroundDark,
    },
    secondary: {
      main: StriveColors.logoPurple,
      dark: StriveColors.logoOutline,
    },
    background: {
      default: StriveColors.backgroundPrimary,
      paper: StriveColors.backgroundDark,
    },
  },
  typography: {
    fontFamily: "Anek Tamil",
    allVariants: {
      color: "white",
    },
  },
  components: {
    MuiDataGrid: {
      styleOverrides: {
        columnHeader: {
          fontSize: 18,
          fontWeight: 500,
          // color: "red",
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          margin: `15px`,
        },
        html: {
          height: `100%`,
        },
      },
    },
    MuiLink: {
      defaultProps: {
        component: LinkBehavior,
      } as LinkProps,
      styleOverrides: {
        root: {
          color: "white",
          fontWeight: "bold",
        },
      },
    },
    MuiButtonBase: {
      defaultProps: {
        LinkComponent: LinkBehavior,
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
