import pytest

from examples.text_chunking import chunk_text
from yasbd import BoundaryDetector


@pytest.mark.parametrize("max_sentences", [1, 2, 3, 8])
@pytest.mark.parametrize("overlap_percent", [0, 0.5, 50, 50.5, 80])
def test_text_chunking_preserves_sentences(max_sentences, overlap_percent):
    sentences = ["Alpha.", " Beta.", " Gamma.", " Delta.", " Epsilon.", " Zeta."]
    chunks = chunk_text("".join(sentences), "en", max_sentences, overlap_percent)
    assert all(chunks)
    assert all(chunk in "".join(sentences) for chunk in chunks)
    assert chunks[0].startswith("Alpha.")
    assert chunks[-1].endswith("Zeta.")
    assert all(sentence.strip() in "".join(chunks) for sentence in sentences)
    detector = BoundaryDetector("en")
    assert all(len(list(detector.segment(chunk))) <= max_sentences for chunk in chunks)


@pytest.mark.parametrize("text", ["Alpha. Beta.", ""])
@pytest.mark.parametrize("overlap_percent", [-50, -0.1, 80.1, 100, 150, float("nan"), float("inf"), -float("inf")])
def test_text_chunking_rejects_invalid_overlap(text, overlap_percent):
    with pytest.raises(ValueError, match="overlap_percent must be between 0 and 80 inclusive"):
        chunk_text(text, "en", 2, overlap_percent)


@pytest.mark.parametrize("text", ["Alpha.", ""])
@pytest.mark.parametrize("max_sentences", [0, -1])
def test_text_chunking_rejects_invalid_window(text, max_sentences):
    with pytest.raises(ValueError, match="max_sentences must be positive"):
        chunk_text(text, "en", max_sentences, 50)


@pytest.mark.parametrize("overlap_percent", [0, 0.5, 50.5, 80])
def test_text_chunking_empty_input(overlap_percent):
    assert chunk_text("", "en", 2, overlap_percent) == []


@pytest.mark.parametrize("overlap_percent, expected", [
    (0, ["Alpha. Beta.", " Gamma. Delta."]),
    (0.5, ["Alpha. Beta.", " Gamma. Delta."]),
    (50, ["Alpha. Beta.", " Beta. Gamma.", " Gamma. Delta."]),
    (50.5, ["Alpha. Beta.", " Beta. Gamma.", " Gamma. Delta."]),
    (80, ["Alpha. Beta.", " Beta. Gamma.", " Gamma. Delta."]),
])
def test_text_chunking_whole_sentence_overlap(overlap_percent, expected):
    assert chunk_text("Alpha. Beta. Gamma. Delta.", "en", 2, overlap_percent) == expected
