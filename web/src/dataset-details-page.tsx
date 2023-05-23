import { Wrapper } from "./components/wrapper";
import { EntityDetailsComponent } from "./components/entity-details-component";

export const DatasetDetailsPage = () => (
  <Wrapper>
    <EntityDetailsComponent entityType="datasets" />
  </Wrapper>
);
