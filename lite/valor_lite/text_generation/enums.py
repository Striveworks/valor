from enum import Enum


class MetricType(str, Enum):
    AnswerCorrectness = "AnswerCorrectness"
    AnswerRelevance = "AnswerRelevance"
    Bias = "Bias"
    BLEU = "BLEU"
    ContextPrecision = "ContextPrecision"
    ContextRecall = "ContextRecall"
    ContextRelevance = "ContextRelevance"
    Faithfulness = "Faithfulness"
    Hallucination = "Hallucination"
    ROUGE = "ROUGE"
    SummaryCoherence = "SummaryCoherence"
    Toxicity = "Toxicity"

    @classmethod
    def text_generation(cls) -> set["MetricType"]:
        """
        MetricTypes for text-generation tasks.
        """
        return {
            cls.AnswerCorrectness,
            cls.AnswerRelevance,
            cls.Bias,
            cls.BLEU,
            cls.ContextPrecision,
            cls.ContextRecall,
            cls.ContextRelevance,
            cls.Faithfulness,
            cls.Hallucination,
            cls.ROUGE,
            cls.SummaryCoherence,
            cls.Toxicity,
        }


class ROUGEType(str, Enum):
    ROUGE1 = "rouge1"
    ROUGE2 = "rouge2"
    ROUGEL = "rougeL"
    ROUGELSUM = "rougeLsum"
