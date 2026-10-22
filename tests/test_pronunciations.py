"""voice_services/tts/pronunciations.py: whole-word, case-insensitive respelling."""
import pytest

from voice_services.tts.pronunciations import Lexicon, load


def test_replaces_whole_words_only_and_ignores_case():
    lex = Lexicon({"AI": "ay eye", "RAG": "rag"})

    assert lex.apply("AI said: ai, RAG, rags, maid.") == "ay eye said: ay eye, rag, rags, maid."


def test_longer_entries_win():
    lex = Lexicon({"API": "ay pee eye", "APIs": "ay pee eyes"})

    assert lex.apply("APIs and API") == "ay pee eyes and ay pee eye"


def test_an_empty_lexicon_leaves_text_alone():
    assert Lexicon({}).apply("hello") == "hello"


def test_load_reads_a_file_per_language(tmp_path):
    (tmp_path / "en.json").write_text('{"ciphertext": "sigh fur text"}', encoding="utf-8")

    assert load(tmp_path)["en"].apply("the ciphertext") == "the sigh fur text"
    assert load(tmp_path / "missing") == {}


def test_load_rejects_a_malformed_file(tmp_path):
    (tmp_path / "en.json").write_text('{"AI": 3}', encoding="utf-8")

    with pytest.raises(ValueError, match="en.json"):
        load(tmp_path)
