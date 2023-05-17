import Drawer from "@mui/material/Drawer";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import Link from "@mui/material/Link";
import Box from "@mui/material/Box";
import React from "react";
import { useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

const drawerWidth = 240;

const SetTokenComponent = () => {
  const { getAccessTokenSilently } = useAuth0();

  useEffect(() => {
    const getToken = async () => {
      const token = await getAccessTokenSilently();
      sessionStorage.setItem("token", token);
    };
    getToken();
  });

  return <></>;
};

export const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <Box sx={{ display: "flex" }}>
    <SetTokenComponent />
    <Drawer
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
        },
      }}
      variant="permanent"
      anchor="left"
    >
      <Toolbar />
      <Divider />
      <List>
        <ListItem>
          <Link href="/">Home</Link>
        </ListItem>
        <ListItem>
          <Link href="/models">Models</Link>
        </ListItem>
        <Divider />
        <ListItem>
          <Link href="/datasets">Datasets</Link>
        </ListItem>
        <ListItem>
          <Link href="/profile">Profile</Link>
        </ListItem>
        <Divider />
        <ListItem>
          <Link href="/docs/" component="a" target="_blank">
            Docs <OpenInNewIcon />
          </Link>
        </ListItem>
      </List>
    </Drawer>
    <Box
      component="main"
      sx={{ flexGrow: 1, bgcolor: "background.default", p: 3 }}
    >
      {children}
    </Box>
  </Box>
);
