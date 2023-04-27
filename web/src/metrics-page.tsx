import axios from "axios";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Typography from "@mui/material/Typography";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import FormControl from "@mui/material/FormControl";
import { EvaluationSetting, Metric, MetricAtIOU } from "./velour-types";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableRow from "@mui/material/TableRow";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import { Wrapper } from "./wrapper";

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
  };

  return (
    <FormControl fullWidth>
      <InputLabel id="select-label">Metric type</InputLabel>
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

const MetricsSection = () => {
  let { name, evalSettingsId } = useParams();
  const [selectedMetricType, setSelectedMetricType] = useState("");
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const metricsWithIds = metrics.map((m, i) => ({ ...m, id: i }));
  const url = `${process.env.REACT_APP_BACKEND_URL}/models/${name}/evaluation-settings/${evalSettingsId}/metrics`;
  useEffect(() => {
    axios.get(url).then((response) => {
      setMetrics(response.data);
    });
  }, [url]);
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

const EvalSettingsTable = ({
  evalSetting,
}: {
  evalSetting: EvaluationSetting | undefined;
}): JSX.Element => {
  return (
    <Table>
      <TableBody>
        <TableRow>
          <TableCell variant="head">Model</TableCell>
          <TableCell>{evalSetting?.model_name}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell variant="head">Dataset</TableCell>
          <TableCell>{evalSetting?.dataset_name}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell variant="head">Dataset task type</TableCell>
          <TableCell>{evalSetting?.dataset_gt_task_type}</TableCell>
        </TableRow>
        <TableRow>
          <TableCell variant="head">Model task type</TableCell>
          <TableCell>{evalSetting?.model_pred_task_type}</TableCell>
        </TableRow>
        {evalSetting?.min_area ? (
          <TableRow>
            <TableCell variant="head">Min object area</TableCell>
            <TableCell>{evalSetting?.min_area}</TableCell>
          </TableRow>
        ) : (
          <></>
        )}
        {evalSetting?.max_area ? (
          <TableRow>
            <TableCell variant="head">Max object area</TableCell>
            <TableCell>{evalSetting?.max_area}</TableCell>
          </TableRow>
        ) : (
          <></>
        )}
      </TableBody>
    </Table>
  );
};

const InfoSection = () => {
  let { evalSettingsId } = useParams();
  const [evalSettings, setEvalSettings] = useState<EvaluationSetting>();
  const url = `${process.env.REACT_APP_BACKEND_URL}/evaluation-settings/${evalSettingsId}`;

  useEffect(() => {
    axios.get(url).then((response) => {
      setEvalSettings(response.data);
    });
  }, [url]);

  return <EvalSettingsTable evalSetting={evalSettings} />;
};

export const MetricsPage = () => (
  <Wrapper>
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4">Metrics</Typography>
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <Paper>
            <InfoSection />
          </Paper>
        </Grid>
        <Grid item xs={8}>
          <Paper>
            <MetricsSection />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  </Wrapper>
);
