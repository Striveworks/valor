import axios from "axios";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import FormControl from "@mui/material/FormControl";

type Metric = {
  type: string;
  parameters: { iou: number; ious: number[] };
  label?: { key: string; value: string };
  value: number;
};

type MetricAtIOU = {
  labelKey?: string;
  labelValue?: string;
  value: number;
  iou: number;
  id: number;
};

const APColumns: GridColDef[] = [
  { field: "labelKey", headerName: "Label Key" },
  { field: "labelValue", headerName: "Label Value" },
  { field: "iou", headerName: "IOU" },
  { field: "value", headerName: "Value" },
];

const mAPColumns: GridColDef[] = [
  { field: "iou", headerName: "IOU" },
  { field: "value", headerName: "Value" },
];

const MetricTypeSelect: React.FC<{
  selectedMetricType: string;
  setSelectedMetricType: React.Dispatch<React.SetStateAction<string>>;
}> = ({ selectedMetricType, setSelectedMetricType }) => {
  const handleChange = (event: SelectChangeEvent) => {
    setSelectedMetricType(event.target.value as string);
    console.log(`metric selected: ${selectedMetricType}`);
  };

  return (
    <FormControl fullWidth>
      <InputLabel id="select-label">Metric</InputLabel>
      <Select
        labelId="select-label"
        id="demo-simple-select"
        value={selectedMetricType}
        label="Metric"
        onChange={handleChange}
      >
        <MenuItem value={"AP"}>AP</MenuItem>
        <MenuItem value={"mAP"}>mAP</MenuItem>
      </Select>
    </FormControl>
  );
};

const Switch = ({
  test,
  children,
}: {
  test: string;
  children: JSX.Element[];
}): JSX.Element => {
  const ret = children.find((child) => {
    return child.props.value === test;
  });
  if (ret === undefined) {
    return <></>;
  }
  return ret;
};

const SwitchElement = ({
  value,
  children,
}: {
  value: string;
  children: JSX.Element;
}): JSX.Element => {
  return children;
};

export const MetricsPage = () => {
  let { name, evalSettingsId } = useParams();
  const [selectedMetricType, setSelectedMetricType] = useState("");
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const metricsWithIds = metrics.map((m, i) => ({ ...m, id: i }));
  useEffect(() => {
    const url = `${process.env.REACT_APP_BACKEND_URL}/models/${name}/evaluation-settings/${evalSettingsId}/metrics`;

    axios.get(url).then((response) => {
      setMetrics(response.data);
    });
  });
  if (!metrics) return null;

  const APs: MetricAtIOU[] = [];
  const mAPs: MetricAtIOU[] = [];

  metricsWithIds.forEach((x) => {
    if (x.type === "AP") {
      APs.push({
        labelKey: x.label?.key,
        labelValue: x.label?.value,
        value: x.value,
        iou: x.parameters.iou,
        id: x.id,
      });
    } else if (x.type === "mAP") {
      mAPs.push({ value: x.value, iou: x.parameters.iou, id: x.id });
    }
  });

  return (
    <>
      <MetricTypeSelect
        selectedMetricType={selectedMetricType}
        setSelectedMetricType={setSelectedMetricType}
      />
      <Switch test={selectedMetricType}>
        <SwitchElement value="AP">
          <DataGrid
            rows={APs}
            columns={APColumns}
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
        </SwitchElement>
        <SwitchElement value="mAP">
          <DataGrid
            rows={mAPs}
            columns={mAPColumns}
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
        </SwitchElement>
      </Switch>
    </>
  );
};
