import numpy as np
from PIL import Image

from models.radfm_qna.run import RadFMQnAAdapter, demo


def test_demo_asks_the_authors_question_on_their_image_and_on_a_blank(monkeypatch, tmp_path, capsys):
    Image.new("RGB", (300, 200), (120, 120, 120)).save(tmp_path / "view1_frontal.jpg")
    adapter, calls = RadFMQnAAdapter(), []
    monkeypatch.setattr(adapter, "load", lambda: None)

    def fake(volume, question, options, max_new_tokens):
        calls.append((volume, question, max_new_tokens))
        return "yes", None

    monkeypatch.setattr(adapter, "_generate", fake)
    demo(adapter, str(tmp_path))
    (image, question, n), (blank, question2, _) = calls
    assert image.shape == blank.shape == (512, 512, 4) and image.any() and not blank.any()
    assert question == question2 == "Can you identify any visible signs of Cardiomegaly in the image?"
    assert n == 200 and image.dtype == np.float32
    assert "DEMO" in capsys.readouterr().out
