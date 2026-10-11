from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adaptive_memory

AdaptiveMemory = adaptive_memory.AdaptiveMemory
FILLER_THRESHOLD = adaptive_memory.FILLER_THRESHOLD


# ── Filler learning ───────────────────────────────────────────────────────────

def test_strips_seed_fillers_above_threshold(tmp_path):
    """Seed fillers seen ≥ FILLER_THRESHOLD times must be stripped."""
    memory = AdaptiveMemory(storage_path=tmp_path / 'profile.json')

    for _ in range(FILLER_THRESHOLD):
        memory.observe('um I think like we should go')

    adapted = memory.adapt_transcript('um I think like we should go')

    assert 'um' not in adapted.lower()
    assert 'like' not in adapted.lower()
    summary = memory.summary()
    assert summary['transcript_count'] == FILLER_THRESHOLD
    assert 'um' in summary['learned_fillers']


def test_content_words_never_enter_learned_fillers(tmp_path):
    """Non-seed words (content vocabulary) must never appear in learned_fillers
    regardless of how many times they are observed."""
    memory = AdaptiveMemory(storage_path=tmp_path / 'profile.json')

    for _ in range(10):          # far above any threshold
        memory.observe('deploy the new feature for the launch today')

    fillers = memory.summary()['learned_fillers']
    for content_word in ('deploy', 'feature', 'launch', 'today', 'new'):
        assert content_word not in fillers, (
            f"Content word '{content_word}' leaked into learned_fillers"
        )


def test_content_words_survive_adapt_transcript(tmp_path):
    """Content words must not be stripped from transcripts — regression guard
    for the original bug where every observed word was unconditionally removed."""
    memory = AdaptiveMemory(storage_path=tmp_path / 'profile.json')

    for _ in range(5):
        memory.observe('deploy the new feature today')

    adapted = memory.adapt_transcript('deploy the new feature today')

    assert 'deploy' in adapted.lower()
    assert 'feature' in adapted.lower()


# ── Normalization ─────────────────────────────────────────────────────────────

def test_normalize_informal_speech():
    """Static normalize() must rewrite common informal contractions."""
    assert 'going to' in AdaptiveMemory.normalize('gonna be great')
    assert 'want to'  in AdaptiveMemory.normalize('I wanna try that')
    assert 'got to'   in AdaptiveMemory.normalize('gotta finish this')
    assert 'kind of'  in AdaptiveMemory.normalize('kinda tired today')
    assert 'let me'   in AdaptiveMemory.normalize('lemme check that')


def test_normalize_applied_in_adapt_transcript(tmp_path):
    """adapt_transcript must apply normalization even when no fillers are active."""
    memory = AdaptiveMemory(storage_path=tmp_path / 'profile.json')
    adapted = memory.adapt_transcript('I wanna finish the report')
    assert 'wanna' not in adapted.lower()
    assert 'want to' in adapted.lower()


# ── Phrase tracking ───────────────────────────────────────────────────────────

def test_common_phrases_are_bigrams(tmp_path):
    """common_phrases must be two-word bigrams, not individual words."""
    memory = AdaptiveMemory(storage_path=tmp_path / 'profile.json')

    for _ in range(3):
        memory.observe('launch the new feature for the team')

    phrases = memory.summary()['common_phrases']
    assert phrases, "common_phrases should not be empty after observations"
    for phrase in phrases:
        words = phrase.split()
        assert len(words) == 2, f"Expected bigram, got: {phrase!r}"


def test_phrases_accumulate_across_sessions(tmp_path):
    """phrase_counts must persist and accumulate between save/load cycles."""
    profile = tmp_path / 'profile.json'

    first = AdaptiveMemory(storage_path=profile)
    for _ in range(3):
        first.observe('product launch ready')
    first.save()

    second = AdaptiveMemory(storage_path=profile)
    second.load()
    counts = second.stats.get('phrase_counts', {})
    assert 'product launch' in counts or 'launch ready' in counts


# ── Persistence ───────────────────────────────────────────────────────────────

def test_persists_between_sessions(tmp_path):
    profile = tmp_path / 'profile.json'
    first = AdaptiveMemory(storage_path=profile)
    first.observe('I really like the new setup')
    first.save()

    second = AdaptiveMemory(storage_path=profile)
    second.load()

    assert second.summary()['transcript_count'] == 1
    assert 'like' in second.summary()['learned_fillers']
