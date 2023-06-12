import { EntityDetailsComponent } from './components/entity-details-component';
import { Wrapper } from './components/wrapper';

export const DatasetDetailsPage = () => (
  <Wrapper>
    <EntityDetailsComponent entityType='datasets' />
  </Wrapper>
);
