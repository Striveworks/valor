import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import axios from 'axios';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { usingAuth } from '../auth';
import { EntityResponse } from '../types';

export const EntityDetailsComponent = ({ entityType }: { entityType: string }) => {
  const { name } = useParams();

  const [entityDetails, setEntityDetails] = useState<EntityResponse>();
  const entityDetailsUrl = `${import.meta.env.VITE_BACKEND_URL}/${entityType}/${name}`;
  useEffect(() => {
    let config = {};
    if (usingAuth()) {
      const token = sessionStorage.getItem('token');
      config = { headers: { Authorization: `Bearer ${token}` } };

      if (token === 'null') {
        console.log('token is null');
      }
    }

    axios.get(entityDetailsUrl, config).then((response) => {
      setEntityDetails(response.data);
    });
  }, [entityDetailsUrl]);

  return (
    <>
      <Typography variant='h2'>{name}</Typography>
      <br />
      <Typography>{entityDetails?.description}</Typography>
      <br />
      <Link href={entityDetails?.href} target='_blank'>
        <Typography>{entityDetails?.href}</Typography>
      </Link>
    </>
  );
};
