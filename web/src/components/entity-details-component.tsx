import axios from 'axios';
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { EntityResponse } from '../types';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import { usingAuth } from '../auth';

export const EntityDetailsComponent = ({
  entityType,
}: {
  entityType: string;
}) => {
  let { name } = useParams();

  const [entityDetails, setEntityDetails] = useState<EntityResponse>();
  const entityDetailsUrl = `${
    import.meta.env.VITE_BACKEND_URL
  }/${entityType}/${name}`;
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
      <Link
        href={entityDetails?.href}
        target='_blank'
      >
        <Typography>{entityDetails?.href}</Typography>
      </Link>
    </>
  );
};
