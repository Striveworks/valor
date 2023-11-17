import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import axios from 'axios';
import { useEffect, useState } from 'react';
import { usingAuth } from '../Auth';
import { EntityResponse } from '../types';
import { Wrapper } from './wrapper';

export const ListingComponent = ({
  name,
  pageTitle
}: {
  name: string;
  pageTitle: string;
}) => {
  const [entities, setEntities] = useState<EntityResponse[]>([]);
  const url = `${import.meta.env.VITE_BACKEND_URL}/${name}`;

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
      setEntities(response.data);
    });
  }, [url]);
  if (!entities) return null;

  return (
    <Wrapper>
      <Typography variant='h2'>{pageTitle}</Typography>
      {entities.map((entity) => (
        <>
          <Link href={`/${name}/${entity.name}`} sx={{ fontSize: 20 }} key={entity.name}>
            {entity.name}
          </Link>
          <br />
        </>
      ))}
    </Wrapper>
  );
};
