import axios from "axios";
import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { EntityResponse } from "../types";
import Link from "@mui/material/Link";
import Typography from "@mui/material/Typography";

export const EntityDetailsComponent = ({
  entityType,
}: {
  entityType: string;
}) => {
  let { name } = useParams();

  const [entityDetails, setEntityDetails] = useState<EntityResponse>();
  const entityDetailsUrl = `${process.env.REACT_APP_BACKEND_URL}/${entityType}/${name}`;
  useEffect(() => {
    axios.get(entityDetailsUrl).then((response) => {
      setEntityDetails(response.data);
    });
  }, [entityDetailsUrl]);

  return (
    <>
      <Typography variant="h2">{name}</Typography>
      <br />
      <Typography>{entityDetails?.description}</Typography>
      <br />
      <Link href={entityDetails?.href} target="_blank">
        <Typography>{entityDetails?.href}</Typography>
      </Link>
    </>
  );
};
