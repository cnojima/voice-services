"""voice_services/tts/pronunciations.py: whole-word, case-insensitive respelling."""
import pytest

from voice_services.tts.pronunciations import Lexicon, load, update


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


def test_update_merges_replaces_case_insensitively_and_removes(tmp_path):
    (tmp_path / "en.json").write_text('{"API": "ay pee eye", "RAG": "rag"}', encoding="utf-8")

    lex = update("en", {"api": "A P I", "GPU": "jee pee you"}, ["rag"], tmp_path)

    assert lex.entries == {"api": "A P I", "GPU": "jee pee you"}
    assert load(tmp_path)["en"].apply("the GPU API") == "the jee pee you A P I"
    assert [p.name for p in tmp_path.iterdir()] == ["en.json"]  # no temp file left behind


def test_update_creates_a_missing_file_and_directory(tmp_path):
    update("ja", {"AI": "エーアイ"}, directory=tmp_path / "new")

    assert load(tmp_path / "new")["ja"].apply("AI") == "エーアイ"


def test_update_rejects_bad_input_without_touching_the_file(tmp_path):
    path = tmp_path / "en.json"
    path.write_text('{"AI": "ay eye"}', encoding="utf-8")

    for bad in ({"": "x"}, {"AI": " "}):
        with pytest.raises(ValueError):
            update("en", bad, directory=tmp_path)
    with pytest.raises(ValueError, match="invalid language"):
        update("../evil", {"a": "b"}, directory=tmp_path)
    assert path.read_text(encoding="utf-8") == '{"AI": "ay eye"}'


def test_update_refuses_a_malformed_existing_file(tmp_path):
    (tmp_path / "en.json").write_text("{oops", encoding="utf-8")

    with pytest.raises(ValueError, match="en.json"):
        update("en", {"a": "b"}, directory=tmp_path)
