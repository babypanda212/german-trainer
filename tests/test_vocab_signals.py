from trainer.vocab_signals import tokenize, match_target_words, compute_vocab_signals, KNOWN_QUALITY, UNKNOWN_QUALITY

def test_tokenize_splits_on_non_letters_and_keeps_umlauts():
    assert tokenize("Ich möchte Frühstück, bitte!") == ["Ich", "möchte", "Frühstück", "bitte"]

def test_match_target_words_is_case_insensitive_and_deduplicates():
    words = match_target_words("Der Bahnhof ist neben dem BAHNHOF.", {"Bahnhof"})
    assert words == ["Bahnhof"]   # canonical casing from target_words, matched once despite two occurrences

def test_match_target_words_does_not_stem_or_lemmatize():
    # "gehen" is the target word; "gegangen" is a different surface form and must NOT match.
    assert match_target_words("Ich bin gestern gegangen.", {"gehen"}) == []

def test_match_target_words_ignores_words_outside_the_target_set():
    assert match_target_words("Ich habe ein Auto und ein Haus.", {"Auto"}) == ["Auto"]

def test_compute_vocab_signals_all_matched_words_pass_when_nothing_asked():
    signals = compute_vocab_signals("Ich fahre zum Bahnhof und kaufe Brot.", {"Bahnhof", "Brot"}, asked_about=None)
    assert signals == {"Bahnhof": KNOWN_QUALITY, "Brot": KNOWN_QUALITY}

def test_compute_vocab_signals_asking_about_one_word_excludes_every_other_word_in_the_sentence():
    # "Bahnhof" and "Brot" both appear, but she asked about "Bahnhof" - "Brot" must get NO signal
    # at all (not even a pass), because she only ever asks about one word at a time.
    signals = compute_vocab_signals("Ich fahre zum Bahnhof und kaufe Brot.", {"Bahnhof", "Brot"}, asked_about="Bahnhof")
    assert signals == {"Bahnhof": UNKNOWN_QUALITY}

def test_compute_vocab_signals_asked_about_word_counts_even_if_not_in_target_words():
    # She can ask about any word, not just ones from the curated lists.
    signals = compute_vocab_signals("Was bedeutet Schmetterling?", set(), asked_about="Schmetterling")
    assert signals == {"Schmetterling": UNKNOWN_QUALITY}

def test_compute_vocab_signals_empty_when_nothing_matches_and_nothing_asked():
    assert compute_vocab_signals("Hallo, wie geht es dir?", {"Bahnhof"}, asked_about=None) == {}
