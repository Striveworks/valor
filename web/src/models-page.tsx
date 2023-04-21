import axios from "axios";
import { useState, useEffect } from "react";

type ModelResponse = {
  name: string;
};

export const ModelsPage = () => {
  const [models, setModels] = useState<ModelResponse[]>([]);
  useEffect(() => {
    const url = `${process.env.REACT_APP_BACKEND_URL}/models`;

    axios.get(url).then((response) => {
      console.log(response);
      setModels(response.data);
    });
  }, []);
  if (!models) return null;

  return (
    <>
      <h2>Models</h2>
      {models.map((model) => (
        <p>{model.name}</p>
      ))}
    </>
  );
};
