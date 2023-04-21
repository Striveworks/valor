import axios from "axios";
import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import Typography from "@mui/material/Typography";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import Link from "@mui/material/Link";

type EvaluationSetting = {
  model_name: string;
  dataset_name: string;
  model_pred_task_type: string;
  dataset_gt_task_type: string;
  min_area: number;
  max_area: number;
  id: number;
};

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

  useEffect(() => {
    const url = `${process.env.REACT_APP_BACKEND_URL}/models/${name}/evaluation-settings`;

    axios.get(url).then((response) => {
      setAllEvalSettings(response.data);
    });
  }, []);
  if (!allEvalSettings) return null;

  return (
    <>
      <Typography variant="h2">{name}</Typography>
      <h3>Evaluations</h3>
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
    </>
  );
};
