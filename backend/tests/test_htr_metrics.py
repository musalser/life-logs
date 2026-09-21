from app.htr.application.metrics import MetricsEvaluator, cer, levenshtein, wer


def test_levenshtein():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "axc") == 1
    assert levenshtein("abc", "") == 3
    assert levenshtein(["a", "b"], ["a", "c", "b"]) == 1


def test_cer():
    assert cer("дед", "дед") == 0.0
    assert cer("дед", "дет") == 1 / 3
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0


def test_wer():
    assert wer("Уж очень дед был похож", "Уж очень дед был похож") == 0.0
    assert wer("Уж очень дед", "Уж очен дед") == 1 / 3
    assert wer("не знаю", "незнаю") == 1.0  # merge counts as sub+del


def test_aggregate_pairs():
    evaluator = MetricsEvaluator()
    result = evaluator.evaluate_pairs([("ab", "ab"), ("cd", "cx")])
    assert result["cer"] == 1 / 4
    assert result["lines"] == 2
