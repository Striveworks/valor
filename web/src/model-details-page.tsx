import axios from "axios";
import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import Typography from "@mui/material/Typography";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import Link from "@mui/material/Link";
import { EntityResponse, EvaluationSetting } from "./types";
import { Wrapper } from "./components/wrapper";

const taskTypeWidth = 250;
const areaWidth = 175;
const columns: GridColDef[] = [
  {
    field: "dataset_name",
    headerName: "Dataset",
    // renderCell: (params) => <a href="google.com">params.row.dataset_name</a>,
  },
  {
    field: "model_pred_task_type",
    headerName: "Model prediction type",
    width: taskTypeWidth,
  },
  {
    field: "dataset_gt_task_type",
    headerName: "Dataset annotation type",
    width: taskTypeWidth,
  },
  {
    field: "min_area",
    headerName: "Min. area of objects",
    width: areaWidth,
  },
  {
    field: "max_area",
    headerName: "Max. area of objects",
    width: areaWidth,
  },
  {
    field: "",
    headerName: "",
    renderCell: (params) => (
      <Link href={`evaluation-settings/${params.row.id}`}>view metrics</Link>
    ),
    sortable: false,
    filterable: false,
    hideable: false,
  },
];

export const ModelDetailsPage = () => {
  let { name } = useParams();
  const [allEvalSettings, setAllEvalSettings] = useState<EvaluationSetting[]>(
    []
  );
  const [modelDetails, setModelDetails] = useState<EntityResponse>();
  const modelDetailsUrl = `${process.env.REACT_APP_BACKEND_URL}/models/${name}`;
  useEffect(() => {
    axios.get(modelDetailsUrl).then((response) => {
      setModelDetails(response.data);
    });
  }, [modelDetailsUrl]);

  const evalSettingsUrl = `${modelDetailsUrl}/evaluation-settings`;
  useEffect(() => {
    axios.get(evalSettingsUrl).then((response) => {
      setAllEvalSettings(response.data);
    });
  }, [evalSettingsUrl]);

  return (
    <Wrapper>
      <Typography variant="h2">{name}</Typography>
      <br />
      <Typography>{modelDetails?.description}</Typography>
      <br />
      <Link href={modelDetails?.href} target="_blank">
        <Typography>{modelDetails?.href}</Typography>
      </Link>
      <br />
      <Typography variant="h4">Evaluations</Typography>
      <DataGrid
        rows={allEvalSettings}
        columns={columns}
        initialState={{
          pagination: {
            paginationModel: {
              pageSize: 5,
            },
          },
        }}
        pageSizeOptions={[5]}
        disableRowSelectionOnClick
      />
    </Wrapper>
  );
};
