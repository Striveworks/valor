import Box from '@mui/material/Box';
import FormControl from '@mui/material/FormControl';
import Grid from '@mui/material/Grid';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import axios from 'axios';
import { ChangeEvent, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { usingAuth } from './auth';
import { Wrapper } from './components/wrapper';
import { EvaluationSetting, Metric } from './types';

const metricColumns: GridColDef[] = [
	{ field: 'label', headerName: 'Label', width: 300 },
	{ field: 'parameters', headerName: 'Parameters', width: 300 },
	{ field: 'value', headerName: 'Value' }
];

const MetricTypeSelect: React.FC<{
	selectedMetricType: string;
	setSelectedMetricType: React.Dispatch<React.SetStateAction<string>>;
	metricTypes: string[];
}> = ({ selectedMetricType, setSelectedMetricType, metricTypes }) => {
	const handleChange = (event: ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
		setSelectedMetricType(event?.target?.value);
	};

	return (
		<FormControl fullWidth>
			<TextField
				select
				id='simple-select'
				value={selectedMetricType}
				label='Metric type'
				onChange={handleChange}
			>
				{metricTypes.map((t) => (
					<MenuItem value={t} key={t}>
						{t}
					</MenuItem>
				))}
			</TextField>
		</FormControl>
	);
};

const MetricsSection = () => {
	const { name, evalSettingsId } = useParams();
	const [selectedMetricType, setSelectedMetricType] = useState('');
	const [metrics, setMetrics] = useState<Metric[]>([]);
	const metricsWithIds = metrics.map((m, i) => ({ ...m, id: i }));
	const url = `${
		import.meta.env.VITE_BACKEND_URL
	}/models/${name}/evaluation-settings/${evalSettingsId}/metrics`;
	useEffect(() => {
		let config = {};
		if (usingAuth()) {
			const token = sessionStorage.getItem('token');
			config = { headers: { Authorization: `Bearer ${token}` } };

			if (token === 'null') {
				console.log('token is null');
			}
		}
		axios.get(url, config).then((response) => {
			setMetrics(response.data);
		});
	}, [url]);
	if (!metrics) return null;

	const stringifyIfObject = (x: unknown) => {
		if (typeof x === 'object') {
			return JSON.stringify(x);
		}
		return x;
	};

	const stringifyObjectValues = (obj: Record<string, unknown>) => {
		Object.keys(obj).forEach((k) => {
			obj[k] = stringifyIfObject(obj[k]);
		});
		return obj;
	};

	// TODO: Follow-up on any typing within this variable
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const metricsByType: { [key: string]: any[] } = metrics.reduce((obj, c) => {
		obj[c.type] = [];
		return obj;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
	}, {} as { [key: string]: any[] });

	metricsWithIds.forEach((m) => {
		metricsByType[m['type']].push(m);
	});

	Object.keys(metricsByType).forEach((metricType) => {
		metricsByType[metricType] = metricsByType[metricType].map(stringifyObjectValues);
	});

	return (
		<>
			<MetricTypeSelect
				selectedMetricType={selectedMetricType}
				setSelectedMetricType={setSelectedMetricType}
				metricTypes={Object.keys(metricsByType)}
			/>
			{metricsByType[selectedMetricType] && (
				<DataGrid
					rows={metricsByType[selectedMetricType]}
					columns={metricColumns}
					initialState={{
						pagination: {
							paginationModel: {
								pageSize: 20
							}
						}
					}}
					pageSizeOptions={[5]}
					disableRowSelectionOnClick
				/>
			)}
		</>
	);
};

const EvalSettingsTable = ({
	evalSetting
}: {
	evalSetting: EvaluationSetting | undefined;
}): JSX.Element => {
	return (
		<Table>
			<TableBody>
				<TableRow>
					<TableCell variant='head'>Model</TableCell>
					<TableCell>{evalSetting?.model_name}</TableCell>
				</TableRow>
				<TableRow>
					<TableCell variant='head'>Dataset</TableCell>
					<TableCell>{evalSetting?.dataset_name}</TableCell>
				</TableRow>
				<TableRow>
					<TableCell variant='head'>Dataset task type</TableCell>
					<TableCell>{evalSetting?.dataset_gt_task_type}</TableCell>
				</TableRow>
				<TableRow>
					<TableCell variant='head'>Model task type</TableCell>
					<TableCell>{evalSetting?.model_pred_task_type}</TableCell>
				</TableRow>
				{evalSetting?.min_area ? (
					<TableRow>
						<TableCell variant='head'>Min object area</TableCell>
						<TableCell>{evalSetting?.min_area}</TableCell>
					</TableRow>
				) : (
					<></>
				)}
				{evalSetting?.max_area ? (
					<TableRow>
						<TableCell variant='head'>Max object area</TableCell>
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
	const { evalSettingsId } = useParams();
	const [evalSettings, setEvalSettings] = useState<EvaluationSetting>();
	const url = `${import.meta.env.VITE_BACKEND_URL}/evaluation-settings/${evalSettingsId}`;

	useEffect(() => {
		let config = {};
		if (usingAuth()) {
			const token = sessionStorage.getItem('token');
			config = { headers: { Authorization: `Bearer ${token}` } };

			if (token === 'null') {
				console.log('token is null');
			}
		}

		axios.get(url, config).then((response) => {
			setEvalSettings(response.data);
		});
	}, [url]);

	return <EvalSettingsTable evalSetting={evalSettings} />;
};

export const MetricsPage = () => (
	<Wrapper>
		<Box sx={{ flexGrow: 1 }}>
			<Typography variant='h4'>Metrics</Typography>
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
