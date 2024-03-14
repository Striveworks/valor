import argparse
import random
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
n_egs = 100
answer_choices = ["A", "B", "C", "D", "E"]
instructions = [
    "Please answer the following question with only the letter of the correct option in the bracket",
    "You are a medical professional, please answer the following question with only the letter of the correct option in the bracket.",
    "You are impersonating the amazing, never wrong doctor from the show House MD, please answer the following question as he would with the letter of the correct option in the bracket.",
    "You are a medical professional who always thinks through and finds justification in your answers, please answer the following question with only the letter of the correct option in the bracket.",
    "Considering the perspective of a patient looking for clear and authoritative advice, please answer the following question with only the letter of the correct option in the bracket.",
    "Channeling the wisdom of a Nobel Prize-winning physician who has significantly contributed to medical science, please answer the following question with only the letter of the correct option in the bracket.",
    "As a clinician renowned for your diagnostic acumen and treatment precision, please answer the following question with only the letter of the correct option in the bracket.",
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
    template = get_prompt_template(model.name)

    return template.format(
        prompt=f"{model.metadata['instruction']}\n{datum.metadata['input']}"
    )


def get_openai_outlines_model(model_name: str) -> models.OpenAI:
    return models.openai(model_name)


def get_llama_cpp_outlines_model(model_path: PosixPath) -> models.LlamaCpp:
    return models.llamacpp(str(model_path), n_ctx=2048)


def get_prompt_template(name: str):
    if "mixtral" in name:
        return "[INST] {prompt} [/INST]"
    if "mistral" in name:
        return "<s>[INST] {prompt} [/INST]"
    if "openai" in name:
        return "{prompt}"
    else:
        raise ValueError(f"unknown model name {name}")


def create_valor_model(
    name: str, instruction: str, valor_model_metadata: dict | None = None
) -> Model:
    valor_model_metadata = valor_model_metadata or {}
    valor_model_metadata["instruction"] = instruction
    model = Model.create(name=name, metadata=valor_model_metadata)
    return model


def maybe_run_inference(
    name: str,
    instruction: str,
    outlines_model: models.OpenAI | models.LlamaCpp,
    dset: Dataset,
    valor_model_metadata: dict | None = None,
):
    model = Model.get(name=name)
    if model is None:
        model = create_valor_model(name, instruction, valor_model_metadata)
    else:
        # check if we already have an evaluation and if so, don't run again
        evals = model.get_evaluations()
        for ev in evals:
            if ev.datum_filter.dataset_names == [dset.name]:
                print(f"evaluation already exists for {name}, skipping")
                return

    print(f"running inference for {name}")
    generator = generate.choice(outlines_model, answer_choices)

    for datum in tqdm(dset.get_datums()):
        if model.get_prediction(dset, datum):
            continue

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--model_files", type=str, required=True)

    args = parser.parse_args()
    base_path = args.base_path
    fnames = args.model_files.split(",")

    connect(args.host)
    client = Client()

    maybe_create_dataset()

    openai_model_name = "gpt-3.5-turbo"

    names_models_and_metadata = [
        (
            openai_model_name,
            get_openai_outlines_model(openai_model_name),
            {"model_name": openai_model_name, "runtime": "openai"},
        )
    ] + [
        (
            fname,
            get_llama_cpp_outlines_model(base_path / fname),
            {"model_file": fname, "runtime": "llama_cpp"},
        )
        for fname in fnames
    ]

    for name, outlines_model, valor_metadata in names_models_and_metadata:
        for i, instruction in enumerate(instructions):
            maybe_run_inference(
                name=f"{name.replace('.', '')}_prompt{i}",
                instruction=instruction,
                outlines_model=outlines_model,
                dset=client.get_dataset(dset_name),
                valor_model_metadata=valor_metadata,
            )
