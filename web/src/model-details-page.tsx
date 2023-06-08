import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import axios from 'axios';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { usingAuth } from './auth';
import { EntityDetailsComponent } from './components/entity-details-component';
import { Wrapper } from './components/wrapper';
import { EvaluationSetting } from './types';

const taskTypeWidth = 250;
const areaWidth = 175;
const columns: GridColDef[] = [
	{
		field: 'dataset_name',
		headerName: 'Dataset',
		width: 150
	},
	{
		field: 'model_pred_task_type',
		headerName: 'Model prediction type',
		width: taskTypeWidth
	},
	{
		field: 'dataset_gt_task_type',
		headerName: 'Dataset annotation type',
		width: taskTypeWidth
	},
	{
		field: 'min_area',
		headerName: 'Min. area of objects',
		width: areaWidth
	},
	{
		field: 'max_area',
		headerName: 'Max. area of objects',
		width: areaWidth
	},
	{
		field: '',
		headerName: '',
		renderCell: (params) => (
			<Link href={`evaluation-settings/${params.row.id}`}>view metrics</Link>
		),
		sortable: false,
		filterable: false,
		hideable: false
	}
];

export const ModelDetailsPage = () => {
	const { name } = useParams();
	const [allEvalSettings, setAllEvalSettings] = useState<EvaluationSetting[]>([]);
	const evalSettingsUrl = `${
		import.meta.env.VITE_BACKEND_URL
	}/models/${name}/evaluation-settings`;
	useEffect(() => {
		let config = {};
		if (usingAuth()) {
			const token = sessionStorage.getItem('token');
			config = { headers: { Authorization: `Bearer ${token}` } };

			if (token === 'null') {
				console.log('token is null');
			}
		}

		axios.get(evalSettingsUrl, config).then((response) => {
			setAllEvalSettings(response.data);
		});
	}, [evalSettingsUrl]);

	return (
		<Wrapper>
			<EntityDetailsComponent entityType='models' />
			<br />
			<Typography variant='h4'>Evaluations</Typography>
			<DataGrid
				rows={allEvalSettings}
				columns={columns}
				initialState={{
					pagination: {
						paginationModel: {
							pageSize: 5
						}
					}
				}}
				pageSizeOptions={[5]}
				disableRowSelectionOnClick
			/>
		</Wrapper>
	);
};
