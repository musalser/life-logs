


# ---------------------------------------------------------------------------
# the dictionary has exactly two sources
# ---------------------------------------------------------------------------


def test_the_knowledge_base_is_not_a_dictionary_source():
    """Entity extraction stores word pieces ("##ang", "Gol", "##eus Grafana").

    They used to reach the author dictionary through the knowledge base, which
    made an author with no confirmed page show a list of debris. The vocabulary
    is the Russian word list plus the author's own confirmed words — nothing else.
    """
    from app.htr.infrastructure.lexicon import LayeredLexiconChecker
    from app.htr.infrastructure.lexicon.corpus import AuthorCorpusSnapshot

    snapshot = AuthorCorpusSnapshot(author_words=frozenset({"шарья"}), author_terms=(("шарья", 2),))
    assert snapshot.known_words == frozenset()
    checker = LayeredLexiconChecker(base=None, author=snapshot.author_words)
    assert checker.verdict("шарья") == "author"
    assert checker.verdict("grafana") == "unknown", "термин знаний словарём не является"
