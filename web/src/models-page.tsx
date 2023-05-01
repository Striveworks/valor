import axios from "axios";
import { useState, useEffect } from "react";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";
import { Wrapper } from "./wrapper";

type ModelResponse = {
  name: string;
};

export const ModelsPage = () => {
  const [models, setModels] = useState<ModelResponse[]>([]);
  const url = `${process.env.REACT_APP_BACKEND_URL}/models`;

  useEffect(() => {
    axios.get(url).then((response) => {
      setModels(response.data);
    });
  }, [url]);
  if (!models) return null;

  return (
    <Wrapper>
      <Typography variant="h2">Models</Typography>
      {models.map((model) => (
        <Link
          href={`/models/${model.name}`}
          sx={{ fontSize: 20 }}
          key={model.name}
        >
          {model.name}
        </Link>
      ))}
    </Wrapper>
  );
};
