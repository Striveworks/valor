import axios from "axios";
import { useState, useEffect } from "react";

type ModelResponse = {
  name: string;
};

export const ModelsPage = () => {
  console.log(process.env.REACT_APP_BACKEND_URL);
  const [models, setModels] = useState<ModelResponse[]>([]);
  useEffect(() => {
    const url = `${process.env.REACT_APP_BACKEND_URL}/models`;
    console.log(`url: ${url}`);
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
