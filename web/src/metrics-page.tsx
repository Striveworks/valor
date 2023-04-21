import axios from "axios";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { DataGrid, GridColDef } from "@mui/x-data-grid";

type Metric = {
  type: string;
  parameters: object;
  label: { key: string; value: string };
  value: number;
};
// type MetricWithId =

const columns: GridColDef[] = [
  { field: "type", headerName: "Type" },
  { field: "value", headerName: "Value" },
  { field: "parameters", headerName: "Parameters" },
];

export const MetricsPage = () => {
  let { name, evalSettingsId } = useParams();
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const metricsWithIds = metrics.map((m, i) => ({ ...m, id: i }));
  useEffect(() => {
    const url = `${process.env.REACT_APP_BACKEND_URL}/models/${name}/evaluation-settings/${evalSettingsId}/metrics`;

    axios.get(url).then((response) => {
      setMetrics(response.data);
    });
  }, []);
  if (!metrics) return null;
  return (
    <>
      <DataGrid
        rows={metricsWithIds}
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
