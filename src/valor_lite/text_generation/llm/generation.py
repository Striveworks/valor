from typing import Any, Callable

from valor_lite.text_generation.llm.exceptions import InvalidLLMResponseError
from valor_lite.text_generation.llm.instructions import (
    format_answer_correctness_verdicts_instruction,
    format_answer_relevance_verdicts_instruction,
    format_bias_verdicts_instruction,
    format_claims_instruction,
    format_context_precision_verdicts_instruction,
    format_context_recall_verdicts_instruction,
    format_context_relevance_verdicts_instruction,
    format_faithfulness_verdicts_instruction,
    format_hallucination_verdicts_instruction,
    format_opinions_instruction,
    format_statements_instruction,
    format_summary_coherence_instruction,
    format_toxicity_verdicts_instruction,
)
from valor_lite.text_generation.llm.integrations import ClientWrapper
from valor_lite.text_generation.llm.utilities import (
    find_first_signed_integer,
    trim_and_load_json,
)
from valor_lite.text_generation.llm.validators import (
    validate_statements,
    validate_verdicts,
)


def _generate(
    client: ClientWrapper,
    messages: list[dict[str, str]],
    keys: set[str],
    validator: Callable,
    allowed_values: set[str] | None = None,
    enforce_length: int | None = None,
) -> dict[str, Any]:
    """
    Query the LLM client.

    Parameters
    ----------
    client : ClientWrapper
        The LLM client.
    messages : list[dict[str, str]]
        A formatted list of commands for the LLM.
    keys : list[str]
        The keys used to extract results from the LLM's response.
    validator : Callable
        Specifies a validator to use on the response.
    allowed_values : set[str], optional
        An optional set of values to restrict the results to.
    enforce_length : int, optional
        An optional integer that enforces the length of the result.
    """
    response = client(messages)
    response = trim_and_load_json(response)
    for key in keys:
        validator(
            response=response,
            key=key,
            allowed_values=allowed_values,
            enforce_length=enforce_length,
        )
    return response


def generate_claims(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of claims from a piece of text, using a call to the LLM API.

    Example Text: "Einstein won the noble prize in 1921 for his discovery of the photoelectric effect."

    Example JSON Response:
    {
        "claims": [
            "Einstein won the noble prize for his discovery of the photoelectric effect.",
            "Einstein won the noble prize in 1921."
        ]
    }

    Parameters
    ----------
    text: str
        The text to extract claims from.

    Returns
    -------
    list[str]
        The list of claims extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_claims_instruction(text=text),
        },
    ]
    response = _generate(
        client=client,
        messages=messages,
        keys={"claims"},
        validator=validate_statements,
    )
    return response["claims"]


def generate_opinions(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of opinions from a piece of text, using a call to the LLM API.

    Example Text: "Although most people live in cities, I like living in the countryside. CNN thinks that the government is not doing enough to combat climate change. Earth is the smallest planet in our solar system."

    Example JSON response:
    {
        "opinions": [
            "I like living in the countryside."
        ]
    }

    Parameters
    ----------
    text: str
        The text to extract opinions from.

    Returns
    -------
    list[str]
        The list of opinions extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_opinions_instruction(text=text),
        },
    ]
    response = _generate(
        client=client,
        messages=messages,
        keys={"opinions"},
        validator=validate_statements,
    )
    return response["opinions"]


def generate_statements(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
) -> list[str]:
    """
    Generate a list of statements from a piece of text, using a call to the LLM API.

    Example Text: "These shoes? All of our shoes have a thirty day return policy and can be returned for a full refund!"

    Example JSON Response:
    {
        "statements": [
            "These shoes?",
            "All of our shoes have a thirty day return policy",
            "All of our shoes can be returned for a full refund"
        ]
    }

    Parameters
    ----------
    text: str
        The text to extract statements from.

    Returns
    -------
    list[str]
        The list of statements extracted from the text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_statements_instruction(text=text),
        },
    ]
    response = _generate(
        client=client,
        messages=messages,
        keys={"statements"},
        validator=validate_statements,
    )
    return response["statements"]


def generate_answer_correctness_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    prediction_statements: list[str],
    groundtruth_statements: list[str],
) -> dict[str, list[str]]:
    """
    Generate lists of true positives, false positives and false negatives, using a call to the LLM API.

    Example Query: What is the boiling point of water?

    Example Prediction Statements: [
        "The boiling point of water is 100 degrees Celsius at sea level",
        "The melting point of water is 0 degrees Celsius!"
    ]

    Example Ground Truth Statements: [
        "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "The boiling point of water can change with altitude."
    ]

    Example JSON response:
    {
        "TP": [
            "The boiling point of water is 100 degrees Celsius at sea level"
        ],
        "FP": [
            "The melting point of water is 0 degrees Celsius!"
        ],
        "FN": [
            "The boiling point of water can change with altitude."
        ]
    }

    Parameters
    ----------
    query: str
        The query that both the prediction and ground truth should be answering.
    prediction_statements: list[str]
        The prediction statements to evaluate.
    groundtruth_statements: list[str]
        The ground truth statements to evaluate.

    Returns
    -------
    dict[str, list[str]]
        A dictionary of true positives, false positives and false negatives.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_answer_correctness_verdicts_instruction(
                query=query,
                prediction_statements=prediction_statements,
                groundtruth_statements=groundtruth_statements,
            ),
        },
    ]
    response = _generate(
        client=client,
        messages=messages,
        keys={"TP", "FP", "FN"},
        validator=validate_statements,
    )

    if len(response["TP"]) + len(response["FP"]) != len(prediction_statements):
        raise InvalidLLMResponseError(
            f"Number of true positives and false positives did not match the number of prediction statements: {response}"
        )

    if len(response["FN"]) > len(groundtruth_statements):
        raise InvalidLLMResponseError(
            f"Number of false negatives exceeded the number of ground truth statements '{len(groundtruth_statements)}': {response}"
        )

    return response


def generate_answer_relevance_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    statements: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of answer relevance verdicts for a list of statements, using a call to the LLM API.

    Example Query: What should I do if there is an earthquake?

    Example Statements: ["Shoes.", "Thanks for asking the question!", "Earthquake frequency varies by region.", "Duck and hide"]

    Example JSON response:
    {
        "verdicts": [
            {
                "analysis": "The 'Shoes.' statement is completely irrelevant to the query, which asks about what to do in the event of an earthquake.",
                "verdict": "no"
            },
            {
                "analysis": "This statement refers to the query but does not answer the question.",
                "verdict": "idk"
            },
            {
                "analysis": "The statement is about earthquakes, but it does not provide any advice. The statement could be used as a supporting point for some advice, though, so the relevance is unclear.",
                "verdict": "idk"
            },
            {
                "analysis": "This statement is an answer to the question and provides relevant advice.",
                "verdict": "yes"
            }
        ]
    }

    Parameters
    ----------
    query: str
        The query to evaluate the statements against.
    statements: list[str]
        The statements to evaluate the validity of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each statement. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_answer_relevance_verdicts_instruction(
                query=query,
                statements=statements,
            ),
        },
    ]
    response = _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no", "idk"},
        enforce_length=len(statements),
        validator=validate_verdicts,
    )
    return response["verdicts"]


def generate_bias_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    opinions: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of bias verdicts for a list of opinions, using a call to the LLM API.

    Example Opinions: [
        "Government meddling in healthcare bloats costs and quashes innovation.",
        "Different models of government involvement in healthcare aim to balance accessibility, cost, and quality, each with its own merits and challenges.",
        "Men and women should have equal opportunities in the workforce."
    ]

    Example JSON:
    {
        "verdicts": [
            {
                "analysis": "The opinion 'Government meddling in healthcare bloats costs and quashes innovation' reveals a political bias, emphasizing negative views on government involvement.",
                "verdict": "yes"
            },
            {
                "analysis": "This opinion doesn't show any political bias.",
                "verdict": "no"
            },
            {
                "analysis": "This opinion in favor of 'equal opportunities in the workforce' for men and women does not demonstrate any gender bias.",
                "verdict": "no"
            },
        ]
    }

    Parameters
    ----------
    opinions: list[str]
        The opinions to evaluate the bias of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each opinion. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_bias_verdicts_instruction(
                opinions=opinions,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(opinions),
        validator=validate_verdicts,
    )["verdicts"]


def generate_context_precision_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    ordered_context_list: list[str],
    groundtruth: str,
) -> list[dict[str, str]]:
    """
    Generate a list of context precision verdicts for an ordered list of contexts,
    using a call to the LLM API.

    The verdict for each context should be 'yes' if the context is relevant to
    produce the ground truth answer to the query. The verdict should be 'no'
    otherwise.

    Example Query: "Who won the Nobel Prize in 1921 and for what?"

    Example Context List: [
        "Einstein won the Nobel Prize for his discovery of the photoelectric effect",
        "Einstein won the Nobel Prize in 1921.",
        "Einstein was born in 1879 in Germany.",
    ]

    Example Ground Truth: "Einstein won the Nobel Prize in 1921 for his discovery of the photoelectric effect."

    Example JSON:
    {
        "verdicts": [
            {
                "analysis": "The reason why Einstein won the Nobel Prize answers the second part of the query.",
                "verdict": "yes"
            },
            {
                "reason": "The context answers who won the prize in 1921.",
                "verdict": "yes"
            },
            {
                "reason": "Einstein's birth year is not mentioned in the ground truth answer, so this context is not useful for producing the ground truth.",
                "verdict": "no"
            }
        ]
    }

    Parameters
    ----------
    query: str
        The query.
    ordered_context_list: list[str]
        The ordered list of contexts. Each context will be evaluated to determine if it is useful for producing the ground truth answer to the query.
    groundtruth: str
        The ground truth answer to the query.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_context_precision_verdicts_instruction(
                query=query,
                ordered_context_list=ordered_context_list,
                groundtruth=groundtruth,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(ordered_context_list),
        validator=validate_verdicts,
    )["verdicts"]


def generate_context_recall_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    context_list: list[str],
    groundtruth_statements: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of context recall verdicts for a list of ground truth statements, using a call to the LLM API.

    The verdict for each ground truth statement should be 'yes' if the ground truth statement is attributable to the context list and 'no' otherwise.

    Example Context List: [
        "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical
            physicist, widely held to be one of the greatest and most influential scientists
            of all time. Best known for developing the theory of relativity, he also made important
            contributions to quantum mechanics, and was thus a central figure in the revolutionary
            reshaping of the scientific understanding of nature that modern physics accomplished
            in the first decades of the twentieth century.",
        "Albert Einstein's mass-energy equivalence formula E = mc2, which arises from relativity theory,
            has been called 'the world's most famous equation'.", "Albert Einstein received the 1921 Nobel
            Prize in Physics 'for his services to theoretical physics, and especially for his discovery of
            the law of the photoelectric effect', a pivotal step in the development of quantum theory.
            His work is also known for its influence on the philosophy of science. In a 1999 poll of 130
            leading physicists worldwide by the British journal Physics World, Einstein was ranked the
            greatest physicist of all time. His intellectual achievements and originality have made Einstein
            synonymous with genius."
        ]

    Example Ground Truth Statements: [
        "Albert Einstein was born on 14 March 1879.",
        "Albert Einstein received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
        "Einstein published 4 papers in 1905.",
        "Einstein moved to Switzerland in 1895."
    ]

    Example JSON:
    {
        "verdicts": [
            {
                "analysis": "The date of birth of Einstein is mentioned clearly in the context.",
                "verdict": "yes"
            },
            {
                "reason": "The statement matches exactly with part of a sentence present in the given context.",
                "verdict": "yes"
            },
            {
                "reason": "There is no mention about papers he wrote in the given context.",
                "verdict": "no"
            },
            {
                "reason": "There is no supporting evidence for a move to Switzerland in the given context.",
                "verdict": "no"
            }
        ]
    }

    Parameters
    ----------
    context_list: list[str]
        The list of contexts to evaluate against.
    groundtruth_statements: str
        A list of statements extracted from the ground truth answer.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each ground truth statement. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_context_recall_verdicts_instruction(
                context_list=context_list,
                groundtruth_statements=groundtruth_statements,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(groundtruth_statements),
        validator=validate_verdicts,
    )["verdicts"]


def generate_context_relevance_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    query: str,
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of context relevance verdicts for a list of contexts, using a call to the LLM API.

    Example Query: "What were some of Einstein's achievements?"

    Example Context List: [
        "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1921. He had a cat.",
        "Einstein was born in 1879 in Germany.",
    ]

    Example JSON:
    {
        "verdicts": [
            {
                "analysis": "Einstein's Nobel Prize and discovery of the photoelectric effect are achievements.",
                "verdict": "yes"
            },
            {
                "analysis": "The year and country of Einstein's birth is irrelevant to the question.",
                "verdict": "no"
            },
        ]
    }

    Parameters
    ----------
    query: str
        The query to evaluate each context against.
    context_list: list[str]
        The ordered list of contexts to evaluate the relevance of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_context_relevance_verdicts_instruction(
                query=query,
                context_list=context_list,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(context_list),
        validator=validate_verdicts,
    )["verdicts"]


def generate_faithfulness_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    claims: list[str],
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of faithfulness verdicts for a list of claims, using a call to the LLM API.

    Example Context List: [
        "Einstein won the Nobel Prize for his discovery of the photoelectric effect. Einstein won the Nobel Prize in 1921.",
        "Einstein was a German Scientist.",
    ]

    Example Claims: [
        "Barack Obama was an American president.",
        "Zurich is a city in London",
        "Einstein won the Nobel Prize for the discovery of the photoelectric effect which may have contributed to his fame.",
        "Einstein won the Nobel Prize in 1922 for his discovery of the photoelectric effect.",
        "Einstein was a Germen chef.",
    ]

    Example JSON response:
    {
        "verdicts": [
            {
                "analysis": "Barack Obama is not mentioned in the context list. Therefore, this claim is not faithful to the context.",
                "verdict": "no"
            },
            {
                "analysis": "Zurich is not mentioned in the context list. Therefore, this claim is not faithful.",
                "verdict": "no"
            },
            {
                "analysis": "Einstein's Nobel Prize is mentioned in the context. The claim and context agree that Einstein won the Nobel Prize for his discovery of the photoelectric effect. Therefore this claim is faithful.",
                "verdict": "yes"
            },
            {
                "analysis": "Einstein's Nobel Prize is mentioned in the context. The context and claim give different years for the Nobel Prize, so the claim contradicts the context. Therefore, this claim is not faithful.",
                "verdict": "no"
            },
            {
                "analysis": "The claim and the context give different occupations for Einstein, so the claim is not faithful to the context.",
                "verdict": "no"
            },
        ]
    }

    Parameters
    ----------
    claims: list[str]
        The claims to evaluate the faithfulness of.
    context_list: list[str]
        The list of contexts to evaluate against.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each claim. Each verdict is a dictionary with one key "verdict".
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_faithfulness_verdicts_instruction(
                claims=claims,
                context_list=context_list,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(claims),
        validator=validate_verdicts,
    )["verdicts"]


def generate_hallucination_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
    context_list: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of hallucination verdicts for a list of contexts, using a call to the LLM API.

    The verdict for each context should be 'yes' if the text contradicts that context. The verdict should be 'no' otherwise.

    Example Context List: [
        "Einstein won the Nobel Prize for his discovery of the photoelectric effect.",
        "Einstein won the Nobel Prize in 1921.",
        "Einstein immigrated to the United States in 1933.",
    ]

    Example Text: "Einstein won the Nobel Prize in 1922 for his discovery of the photoelectric effect."

    Example JSON:
    {
        "verdicts": [
            {
                "analysis": "Both the text and the context agree that Einstein won the Nobel Prize for his discovery of the photoelectric effect.",
                "verdict": "no"
            },
            {
                "analysis": "The context states that Einstein won the Nobel Prize in 1921, but the text claims Einstein won the Nobel Prize in 1922.",
                "verdict": "yes"
            },
            {
                "analysis": "The text is unrelated to Einstein immigrating to the U.S., so the text does not contradict this context.",
                "verdict": "no"
            }
        ]
    }

    Parameters
    ----------
    text: str
        The text to evaluate for hallucination.
    context_list: list[str]
        The list of contexts to compare against.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each context. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_hallucination_verdicts_instruction(
                text=text,
                context_list=context_list,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(context_list),
        validator=validate_verdicts,
    )["verdicts"]


def generate_summary_coherence(
    client: ClientWrapper,
    system_prompt: str,
    text: str,
    summary: str,
) -> int:
    """
    Compute summary coherence, the collective quality of a summary.

    Parameters
    ----------
    text: str
        The text that was summarized.
    summary: str
        The summary to be evaluated.

    Returns
    -------
    int
        The summary coherence score will be evaluated as an integer, with 1 indicating the lowest summary coherence and 5 the highest summary coherence.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_summary_coherence_instruction(
                text=text, summary=summary
            ),
        },
    ]

    response = client(messages)

    ret = find_first_signed_integer(response)
    if ret is None:
        raise InvalidLLMResponseError(
            f"LLM response was not a valid summary coherence score: {response}"
        )
    if ret not in {1, 2, 3, 4, 5}:
        raise InvalidLLMResponseError(
            f"Summary coherence score was not an integer between 1 and 5: {ret}"
        )
    return ret


def generate_toxicity_verdicts(
    client: ClientWrapper,
    system_prompt: str,
    opinions: list[str],
) -> list[dict[str, str]]:
    """
    Generate a list of toxicity verdicts for a list of opinions, using a call to the LLM API.

    Example Opinions: [
        "That vegan garbage place is an insult to real food. Only pretentious losers would waste their money on tasteless weeds.",
        "I'm not personally interested in trying the new vegan restaurant, as I prefer dishes with meat.",
        "Sarah always meant well, but you couldn't help but sigh when she volunteered for a project.",
    ]

    Example JSON response:
    {
        "verdicts": [
            {
                "analysis": "The opinion uses words like 'garbage' and 'tasteless weeds', ridicules vegan food and those who choose it, and attacks people who enjoy vegan food, calling them 'pretentious losers'. Futhermore, it dismisses the entire concept of vegan cuisine with a sweeping generalization, making it seem worthless.",
                "verdict": "yes"
            },
            {
                "analysis": "The opinion is respectful and does not attack anyone or anything. It simply states a personal preference.",
                "verdict": "no"
            },
            {
                "analysis": "'Sarah always meant well' sounds positive but is undermined by the surrounding criticism such as 'can't help but sign', which can be considered a personal attack.",
                "verdict": "yes"
            }
        ]
    }

    Parameters
    ----------
    opinions: list[str]
        The opinions to evaluate the toxicity of.

    Returns
    -------
    list[dict[str,str]]
        The list of verdicts for each opinion. Each verdict is a dictionary with the "verdict" field.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": format_toxicity_verdicts_instruction(
                opinions=opinions,
            ),
        },
    ]
    return _generate(
        client=client,
        messages=messages,
        keys={"verdicts"},
        allowed_values={"yes", "no"},
        enforce_length=len(opinions),
        validator=validate_verdicts,
    )["verdicts"]
