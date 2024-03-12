import random
import sys
import time
from pathlib import Path, PosixPath

from datasets import load_dataset
from outlines import generate, models
from tqdm import tqdm

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
    connect,
    enums,
)

dset_name = "MedQA"
n_egs = 10
answer_choices = ["A", "B", "C", "D", "E"]
instructions = [
    "Please answer with only the letter of the correct option in the bracket",
    "You are a medical professional, please answer the following question with only the letter of the correct option in the bracket.",
    "You are impersonating the amazing, never wrong doctor from the show House MD, please answer the following question as he would with the letter of the correct option in the bracket.",
]


def maybe_create_dataset():
    # get a random sample of 1000 from the dataset
    if client.get_dataset(dset_name):
        print("dataset already exists, skipping")
        return
    dataset = load_dataset("medalpaca/medical_meadow_medqa")["train"]
    random.seed(12)
    indices = random.sample(range(len(dataset)), n_egs)

    dset = Dataset.create(dset_name)

    gts = [
        GroundTruth(
            datum=Datum(
                uid=str(i),
                metadata={"input": dataset[i]["input"]},
            ),
            annotations=[
                Annotation(
                    labels=[
                        Label(key="answer", value=dataset[i]["output"][0])
                    ],
                    metadata={"full_answer": dataset[i]["output"]},
                    task_type=enums.TaskType.CLASSIFICATION,
                )
            ],
        )
        for i in indices
    ]

    print("adding groundtruths")
    start = time.perf_counter()
    dset.add_groundtruths(gts)
    print(
        f"finished adding groundtruths in {time.perf_counter() - start:.2f} seconds"
    )
    dset.finalize()


def get_prompt(model: Model, datum: Datum) -> str:
    return f"{model.metadata['instruction']}\n{datum.metadata['input']}"


def get_openai_outlines_model(model_name: str) -> models.OpenAI:
    return models.openai(model_name)


def get_llama_cpp_outlines_model(model_path: PosixPath) -> models.LlamaCpp:
    return models.llamacpp(str(model_path))


def maybe_run_inference(
    name: str,
    instruction: str,
    outlines_model: models.OpenAI | models.LlamaCpp,
    dset: Dataset,
    valor_model_metadata: dict | None = None,
):
    model = Model.get(name=name)
    if model is not None:
        print("model already exists, skipping inference")
        return

    valor_model_metadata = valor_model_metadata or {}
    valor_model_metadata["instruction"] = instruction
    model = Model.create(name=name, metadata=valor_model_metadata)

    generator = generate.choice(outlines_model, answer_choices)

    for datum in tqdm(dset.get_datums()):
        answer = generator(get_prompt(model, datum))

        model.add_prediction(
            dset,
            prediction=Prediction(
                datum=datum,
                annotations=[
                    Annotation(
                        labels=[Label(key="answer", value=answer, score=1.0)],
                        task_type=enums.TaskType.CLASSIFICATION,
                    )
                ],
            ),
        )

    model.evaluate_classification(datasets=dset).wait_for_completion()


base_path = Path("/Users/eric/Downloads/")


def delete_everything():
    y_or_n = input("Are you sure you want to delete everything? (y/n): ")
    if y_or_n.lower() == "y":
        for obj in client.get_models() + client.get_datasets():
            obj.delete()


if __name__ == "__main__":
    connect(
        "https://valor.striveworks.com/api/v1",
        username="user",
        password="bkhiRuzcIfUT",
    )

    client = Client()

    if len(sys.argv) > 1 and sys.argv[1] == "delete":
        delete_everything()
        sys.exit()

    maybe_create_dataset()

    fname = "dolphin-2.1-mistral-7b.Q4_K_M.gguf"
    openai_model_name = "gpt-3.5-turbo"
    for i, instruction in enumerate(instructions):
        print("Running inference through OpenAI")
        maybe_run_inference(
            name=f"{openai_model_name.replace('.', '')}_prompt{i}",
            instruction=instruction,
            outlines_model=get_openai_outlines_model(openai_model_name),
            dset=client.get_dataset(dset_name),
            valor_model_metadata={
                "model_name": openai_model_name,
                "runtime": "openai",
            },
        )

        print("Running inference through LlamaCpp")
        maybe_run_inference(
            name=f"{fname.replace('.', '')}_prompt{i}",
            instruction=instruction,
            outlines_model=get_llama_cpp_outlines_model(base_path / fname),
            dset=client.get_dataset(dset_name),
            valor_model_metadata={"model_file": fname, "runtime": "llama_cpp"},
        )
