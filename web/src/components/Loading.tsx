import { CircularProgress } from '@mui/material';
import { Box } from '@striveworks/minerva';

export function Loading() {
  return (
    <Box xcss={{ justifyContent: 'center', alignItems: 'center' }}>
      <CircularProgress color='secondary' />
    </Box>
  );
}
