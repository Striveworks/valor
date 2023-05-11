import axios from "axios";
import { useState, useEffect } from "react";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";
import { Wrapper } from "./wrapper";

export const ListingComponent = ({
  name,
  pageTitle,
}: {
  name: string;
  pageTitle: string;
}) => {
  const [entities, setEntities] = useState<{ name: string }[]>([]);
  const url = `${process.env.REACT_APP_BACKEND_URL}/${name}`;

  useEffect(() => {
    axios.get(url).then((response) => {
      setEntities(response.data);
    });
  }, [url]);
  if (!entities) return null;

  return (
    <Wrapper>
      <Typography variant="h2">{pageTitle}</Typography>
      {entities.map((entity) => (
        <Link
          href={`/${name}/${entity.name}`}
          sx={{ fontSize: 20 }}
          key={entity.name}
        >
          {entity.name}
        </Link>
      ))}
    </Wrapper>
  );
};
