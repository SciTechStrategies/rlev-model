from collections import defaultdict

PUNCT = '()-:;\'",.{}[]?'


def get_words(text, punct=PUNCT):
    """
    Get set of words in text.

    text is lowercased, and all punctuation removed.
    """
    text = text.lower()
    for p in PUNCT:
        text = text.replace(p, ' ')
    return set([w for w in text.split(' ') if w])


def find_word_features(words, features):
    """
    For the given set of words and word->id map, return set of ids.

    If word in words is found in features, returned set should contain
    that feature id.
    """
    return set(features[w] for w in words if w in features)


def gen_pid_words(pid_text):
    """
    For the given sequence of pid, text pairs, yield pid, words.
    """
    for pid, text in pid_text:
        yield pid, get_words(text)


def gen_pid_word_features(pid_words, features):
    """
    For the given sequence of pid, words pairs, yield pid, features.
    """
    for pid, words in pid_words:
        yield pid, find_word_features(words, features)


def consolidate_pid_word_features(pid_features):
    """
    Returns a dict from pid to feature ids including ids from all sequences.
    """
    pf = defaultdict(set)
    for sequence in pid_features:
        for pid, features in sequence:
            pf[pid].update(features)
    return pf


def get_pid_abstr_title_features(
        pid_abstr, abstr_features, pid_title, title_features
        ):
    """
    Returns map from pid to set of feature ids.

    pid_abstr is a source of pid, abstract text.
    abstr_features is a map from words to feature ids
    pid_title is a source of pid, title text.
    title_features is a map from words to feature ids
    """
    pid_abstr_feat = gen_pid_word_features(
        gen_pid_words(pid_abstr), abstr_features
    )
    pid_title_feat = gen_pid_word_features(
        gen_pid_words(pid_title), title_features
    )
    return consolidate_pid_word_features([
        pid_abstr_feat, pid_title_feat
    ])


def test_get_words():
    """ Test the get_words function. """
    text = "This is Some (Good) Text.  Or is this good text?"
    words = get_words(text)
    assert len(words) == 6
    assert 'this' in words
    assert 'is' in words
    assert 'some' in words
    assert 'good' in words
    assert 'text' in words
    assert 'or' in words


def test_find_word_features():
    """ Test the get_word_features function. """
    features = {"excellent": 0, "good": 1, "great": 2, "ok": 3}
    words = set([
        "this", "is", "some", "good", "text", "or", "just", "ok",
    ])
    word_features = find_word_features(words, features)
    assert len(word_features) == 2
    assert 1 in word_features
    assert 3 in word_features


def test_gen_pid_words():
    """ Test the gen_pid_words function. """
    pid_text = [
        [12345, "This is Some ok Text.  Or is it good text?"],
        [23456, "Better text comes next.  Maybe not."],
    ]
    pid_words = [x for x in gen_pid_words(pid_text)]
    assert len(pid_words) == 2
    assert len(pid_words[0]) == 2
    assert pid_words[0][0] == 12345
    assert len(pid_words[0][1]) == 8
    assert "this" in pid_words[0][1]
    assert "is" in pid_words[0][1]
    assert "some" in pid_words[0][1]
    assert "ok" in pid_words[0][1]
    assert "text" in pid_words[0][1]
    assert "or" in pid_words[0][1]
    assert "it" in pid_words[0][1]
    assert "good" in pid_words[0][1]
    assert len(pid_words[1]) == 2
    assert pid_words[1][0] == 23456
    assert len(pid_words[1][1]) == 6
    assert "better" in pid_words[1][1]
    assert "text" in pid_words[1][1]
    assert "comes" in pid_words[1][1]
    assert "next" in pid_words[1][1]
    assert "maybe" in pid_words[1][1]
    assert "not" in pid_words[1][1]


def test_gen_pid_word_features():
    """ Test the gen_pid_word_features function. """
    pid_words = [
        [12345, set([
            "this", "is", "some", "ok", "text", "or", "it", "good"
        ])],
        [23456, set(["better", "text", "comes", "next", "maybe", "not"])],
    ]
    features = {"excellent": 0, "good": 1, "great": 2, "ok": 3}
    pid_features = [x for x in gen_pid_word_features(pid_words, features)]
    assert len(pid_features) == 2
    assert len(pid_features[0]) == 2
    assert pid_features[0][0] == 12345
    assert len(pid_features[0][1]) == 2
    assert 1 in pid_features[0][1]
    assert 3 in pid_features[0][1]
    assert len(pid_features[1]) == 2
    assert pid_features[1][0] == 23456
    assert len(pid_features[1][1]) == 0


def test_consolidate_pid_word_features():
    pid_features_1 = [
        [12345, set([1, 2, 3])],
        [23456, set([2, 3])],
    ]
    pid_features_2 = [
        [12345, set([1, 5])],
        [23456, set()],
        [34567, set([3, 4])],
    ]
    result = consolidate_pid_word_features([pid_features_1, pid_features_2])
    assert len(result) == 3
    assert result[12345] == set([1, 2, 3, 5])
    assert result[23456] == set([2, 3])
    assert result[34567] == set([3, 4])


def test_get_pid_abstr_title_features():
    """
    Test the get_pid_abstr_title_features function.
    """
    pid_abstr = [
        [12345, "My abstract it is good. It is great. OK?"],
        [23456, "My abstract is great. it is also ok."],
    ]
    abstr_features = {
        "good": 1,
        "great": 2,
        "ok": 3,
    }
    pid_title = [
        [12345, "This title is really nice."],
        [23456, "I think this title smells funny"],
        [34567, "What an excellent title.  Great stuff."],
    ]
    title_features = {
        "really": 4,
        "excellent": 5,
        "great": 6,
        "nice": 7,
    }
    result = get_pid_abstr_title_features(
        pid_abstr, abstr_features, pid_title, title_features
    )
    assert len(result) == 3
    assert result[12345] == set([1, 2, 3, 4, 7])
    assert result[23456] == set([2, 3])
    assert result[34567] == set([5, 6])
