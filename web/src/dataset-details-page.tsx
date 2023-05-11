import axios from "axios";
import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import Typography from "@mui/material/Typography";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import Link from "@mui/material/Link";
import { EntityResponse, EvaluationSetting } from "./types";
import { Wrapper } from "./components/wrapper";
import { EntityDetailsComponent } from "./components/entity-details-component";

export const DatasetDetailsPage = () => (
  <Wrapper>
    <EntityDetailsComponent entityType="datasets" />
  </Wrapper>
);
