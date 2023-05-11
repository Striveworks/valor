import Drawer from "@mui/material/Drawer";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import Link from "@mui/material/Link";
import Box from "@mui/material/Box";
import React from "react";

const drawerWidth = 240;

export const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <Box sx={{ display: "flex" }}>
    {" "}
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
        <ListItem>
          <Link href="/datasets">Datasets</Link>
        </ListItem>
        <ListItem>
          <Link href="/profile">Profile</Link>
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
