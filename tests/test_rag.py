import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_case():
    answer_relevant_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What are the CVAP areas",
        actual_output="There are 5 areas",
        retrieval_context=[
            "There are 3 areas"
        ]
    )
    assert_test(test_case, [answer_relevant_metric])