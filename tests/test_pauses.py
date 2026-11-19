import pytest

from voice_services.tts import pauses


@pytest.mark.parametrize("text, lang, expected", [
    ("Wait — what? Fine: go; now.", "en", "Wait, what? Fine, go, now."),
    ("one–two and a--b", "en", "one, two and a, b"),
    ("a, — b", "en", "a, b"),
    ("It's 10:30 at https://x.io — ok", "en", "It's 10:30 at https://x.io, ok"),
    ("Items 3: 1:2 ratio", "en", "Items 3, 1:2 ratio"),
    ("待って―はい：行く；今", "ja", "待って、はい、行く、今"),
    ("no marks, here.", "en", "no marks, here."),
    ("Il est 10:30 : voilà !", "fr", "Il est 10:30, voilà !"),
    ("Il dit « bonjour » à tous.", "fr", "Il dit bonjour à tous."),
    ("Attention ; c'est chaud — vraiment ?", "fr", "Attention, c'est chaud, vraiment ?"),
    ("L'uomo c'è; va via: subito", "it", "L'uomo c'è, va via, subito"),
])
def test_pauses_become_commas(text, lang, expected):
    assert pauses.apply(text, lang) == expected
