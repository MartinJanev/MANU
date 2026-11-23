"""parse_choice on chat models that explain their choice (MedGemma's dev answers of 23 Nov, verbatim)."""
from common.parse import parse_choice

SEQ = ["T1-weighted", "T1-weighted with contrast", "T2-weighted", "FLAIR"]
DIAG = ["Normal", "Tumor", "Stroke", "Multiple sclerosis"]
ORIGIN = ["Lung", "Breast", "Melanoma", "Other"]


def test_bold_letter_and_option():
    text = ("Based on the image characteristics, the most likely sequence is **C. T2-weighted**.\n\nHere's why:\n\n"
            "*   **T2-weighted images** typically show fluid (like cerebrospinal fluid) as bright (hyperintense) and "
            "fat as dark (hypointense). While this image doesn't show fat, ... *   **T1-we")
    assert parse_choice(text, SEQ, "A") == 2


def test_letter_follows_the_displayed_order():
    text = "Based on the image characteristics, the most likely sequence is **B. T2-weighted**.\n\nHere's why:"
    assert parse_choice(text, list(reversed(SEQ)), "A") == 1


def test_first_bold_option_beats_the_explanation_of_the_others():
    text = ("Based on the provided brain MRI images, the main finding appears to be **Multiple sclerosis**.\n\n"
            "Here's why:\n\n*   **Multiple Sclerosis (MS):** The images show multiple lesions ...\n"
            "*   **Stroke:** Stroke typically presents as a single, well-defined ar")
    assert parse_choice(text, DIAG, "A") == 3


def test_lung_in_bold():
    text = ("Based on the typical distribution of brain metastases, the most likely primary cancer is **A. Lung**.\n\n"
            "Lung cancer is the most common source of brain metastases, followed by breast cancer. Melanoma is less "
            "common but can also metastasize to the brain.\n")
    assert parse_choice(text, ORIGIN, "A") == 0


def test_letter_and_text_that_disagree_are_no_answer():
    assert parse_choice("The answer is **B. Stroke**.", DIAG, "A") is None


def test_a_repeated_question_is_still_no_answer():
    echo = "Choices: A. Normal B. Tumor C. Stroke D. Multiple sclerosis"
    assert parse_choice(echo, DIAG, "A") is None
