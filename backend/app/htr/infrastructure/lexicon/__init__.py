"""Dictionary (lexicon) infrastructure for the vocabulary check.

* ``bloom`` — the on-disk Bloom filter artifact and its file format;
* ``file_lexicon`` — process-wide loading of that artifact;
* ``corpus`` — the author's own words (confirmed pages + knowledge base);
* ``checker`` — the layered check (general dictionary + author vocabulary);
* ``provider`` — wiring used by the factory.
"""
from .bloom import BloomFilter
from .checker import LayeredLexiconChecker
from .corpus import (
    AuthorCorpusCache,
    AuthorCorpusSnapshot,
    SqlAlchemyAuthorCorpus,
    author_corpus_cache,
)
from .file_lexicon import FileLexicon, clear_lexicon_cache, load_file_lexicon
from .provider import SqlAlchemyLexiconProvider
from .words import words_of

__all__ = [
    "BloomFilter",
    "LayeredLexiconChecker",
    "AuthorCorpusCache",
    "AuthorCorpusSnapshot",
    "SqlAlchemyAuthorCorpus",
    "author_corpus_cache",
    "FileLexicon",
    "load_file_lexicon",
    "clear_lexicon_cache",
    "SqlAlchemyLexiconProvider",
    "words_of",
]
