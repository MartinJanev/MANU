"""The paper's experiments: Q&A and prior knowledge on the full-prompt models, and the Q&A fixes they need."""
import json
import zlib

import nibabel as nib
import numpy as np
import pytest

from common import runner
from common.datasets import Scan
from common.parse import parse_choice
from common.qna import BANK, interrogate
from common.schema import full_prompt
from models.fleming8b.run import FlemingAdapter
from models.healthgptpro27b.run import HealthGPTProAdapter
from models.lingshu7b.run import LingshuAdapter
from models.medgemma15.run import MedGemma15Adapter
from models.qoqmed7b.run import QoQMedAdapter
from models.qwen3vl8b.run import Qwen3VLAdapter

QUESTION = "What is the modality of this image? Choices: A. MRI B. CT C. X-ray D. Ultrasound"


def volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def scan(dataset="molab", prior=False):
    return Scan(dataset, "p1", "p1.nii.gz", ("p1",), prior)


# parse_choice: option texts inside one another

def test_abnormal_is_not_read_as_normal():
    assert parse_choice("The image is abnormal", ["Normal", "Tumor", "Stroke", "Multiple sclerosis"], "A") is None
    assert parse_choice("Abnormal", ["Normal", "Abnormal", "Cannot be determined", "Other"], "A") == 1


def test_the_longer_of_two_nested_options_wins():
    options = ["T1-weighted", "T1-weighted with contrast", "T2-weighted", "FLAIR"]
    assert parse_choice("T1-weighted with contrast", options, "A") == 1
    assert parse_choice("T1-weighted", options, "A") == 0


# option order and the chest question

def test_first_option_order_follows_the_patient():
    ids = [f"p{i}" for i in range(20)]
    odd = next(p for p in ids if zlib.crc32(p.encode()) % 2)
    even = next(p for p in ids if not zlib.crc32(p.encode()) % 2)

    def always_a(prompt):
        return "A.", None

    assert interrogate("ixi", always_a, "A", even)[0]["diagnosis_name"] == "normal"
    assert interrogate("ixi", always_a, "A", odd)[0]["diagnosis_name"] == "multiple sclerosis"


def test_ct_rate_is_asked_normal_or_abnormal():
    asked = []

    def fake(prompt):
        asked.append(prompt)
        return ("Abnormal" if "normal or abnormal" in prompt else "A."), None

    assert interrogate("ct_rate", fake, "A")[0]["diagnosis_name"] == "other abnormalities"
    assert not any("brain image" in p for p in asked)
    assert BANK["diagnosis_ct"].values[0] == "normal"


# prior knowledge

@pytest.mark.parametrize("dataset,told", [("molab", "metastasized tumor"),
                                          ("ms", "This is a scan of a patient with multiple sclerosis.")])
def test_prior_adds_only_the_sentence(dataset, told):
    base, prior = full_prompt(dataset).split("\n"), full_prompt(dataset, prior=True).split("\n")
    assert prior[:1] + prior[2:] == base and told in prior[1]


# every full-prompt adapter: prior reaches the model, and one question can be asked

CASES = [(HealthGPTProAdapter, lambda: ("A. MRI", False, 4)), (Qwen3VLAdapter, lambda: "A. MRI"),
         (LingshuAdapter, lambda: "A. MRI"), (QoQMedAdapter, lambda: "<think>hm</think>A. MRI"),
         (FlemingAdapter, lambda: "<think>hm</think><answer>A. MRI</answer>"), (MedGemma15Adapter, lambda: "A. MRI")]


@pytest.mark.parametrize("cls,output", CASES, ids=[c.name for c, _ in CASES])
def test_asker_returns_the_answer_to_one_question(monkeypatch, cls, output):
    adapter, sent = cls(), []

    def fake(*args):
        sent.append(str(args))
        return output()

    monkeypatch.setattr(adapter, "_generate", fake)
    raw, probs = adapter.asker(scan(), [volume()])(QUESTION)
    assert (raw.strip(), probs) == ("A. MRI", None)
    assert QUESTION in sent[0] and "JSON" not in sent[0]


@pytest.mark.parametrize("cls,output", CASES, ids=[c.name for c, _ in CASES])
def test_predict_passes_the_prior_on(monkeypatch, cls, output):
    adapter, sent = cls(), []

    def fake(*args):
        sent.append(str(args))
        answer = '{"diagnosis_name": "tumor", "modality": "MRI"}'
        return (answer, False, 9) if isinstance(output(), tuple) else answer

    monkeypatch.setattr(adapter, "_generate", fake)
    adapter.predict(scan(prior=True), [volume()])
    adapter.predict(scan(prior=False), [volume()])
    assert "metastasized tumor" in sent[0] and "metastasized tumor" not in sent[1]


# runner --task

class FullPromptModel:
    name = "fake"
    model_id = "fake/model"
    mode = "full_json"

    def __init__(self):
        self.priors = []

    def load(self):
        pass

    def predict(self, scan, volumes):
        self.priors.append(scan.prior)
        return {"diagnosis_name": "tumor"}, {"raw": "x"}

    def asker(self, scan, volumes):
        return lambda question: ("A.", None)


class ClassifierModel:
    name = "clf"
    model_id = "clf/model"
    mode = "clip_or_heads"

    def load(self):
        pass

    def predict(self, scan, volumes):
        return {}, {}


@pytest.fixture
def run_task(tmp_path):
    path = tmp_path / "data/molab/nifti/10005_19010618_eTRA-3D-T1Gd_img.nii"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(path))
    reference = str(runner.Path(__file__).resolve().parent.parent / "reference")

    def go(adapter, task, datasets=("molab",)):
        out = runner.run(adapter, "paper", list(datasets), str(tmp_path / "data"), reference,
                         str(tmp_path / "results"), {"paper": {"molab": ["010005N"]}}, controls=True, task=task)
        (processed,) = out.glob("*_processed.jsonl")
        return out, [json.loads(line) for line in open(processed)]
    return go


def test_qna_task_gets_its_own_results_folder(run_task):
    out, (record,) = run_task(FullPromptModel(), "qna")
    assert out.parent.name == "fake-qna"
    assert record["input"]["system_prompt_type"] == "qna_per_field_v1"
    assert record["output"]["parsed_response"]["modality"] is not None


def test_prior_task_marks_every_scan_and_the_control(run_task):
    adapter = FullPromptModel()
    out, (record,) = run_task(adapter, "prior")
    assert out.parent.name == "fake-prior" and adapter.priors == [True, True]
    assert record["input"]["system_prompt_type"] == "full_json_prior_v1"


def test_the_native_task_keeps_the_plain_name(run_task):
    adapter = FullPromptModel()
    out, _ = run_task(adapter, "full")
    assert out.parent.name == "fake" and adapter.priors == [False, False]


def test_tasks_a_model_cannot_do_are_refused(run_task):
    with pytest.raises(ValueError):
        run_task(ClassifierModel(), "qna")
    with pytest.raises(ValueError):
        run_task(FullPromptModel(), "prior", datasets=("ixi",))
