"""emoji_service.py — Advanced Emotion-Based Contextual Emoji Insertion Engine.

A production-grade, toggleable feature (OFF by default) that analyzes emotional
tone, valence, intensity, and sentiment negation from cleaned, punctuated text,
then appends contextually fitting emojis with syntactic precision.

Core Capabilities:
1. STRICT GUARDRAIL: Disabled by default (`enabled=False`). Unchanged pass-through when OFF.
2. 14 GRANULAR EMOTIONS: Celebratory, Excited, Happy, Grateful, Relieved, Humorous,
   Supportive, Thoughtful, Agreement, Cool, Sad, Frustrated, Apologetic, Urgent.
3. MULTI-LANGUAGE & CULTURAL AWARENESS:
   - Full English idioms and colloquialisms.
   - Deep Pakistani Roman Urdu lexicon and street slang ("phaar diya", "chha gaye", "scenes on hain").
   - Native Urdu script recognition (Nastaliq/Arabic Unicode range: مبارک, زبردست, شکر ہے, etc.).
4. NEGATION & SENTIMENT INVERSION: Prevents false-positives ("not happy", "never excited",
   "acha nahi hai"). Inverts valence or suppresses positive emojis.
5. SERIOUS & SOMBER SUPPRESSION: Respectfully suppresses emojis for grief, bereavement,
   funerals, hospital/ICU emergencies, legal issues, or financial distress.
6. TECHNICAL & INSTRUCTIONAL SUPPRESSION: Detects code, devops, database, and system
   instructions to keep developer workflows clean and professional.
7. INTENSITY & AROUSAL SCORING: Detects intensifiers ("super", "bohot", exclamation marks, caps)
   to adapt emoji selection (e.g., mild appreciation vs. explosive celebration).
8. MULTI-SENTENCE COHESION: Accurately targets the emotional sentence/clause instead of
   blindly appending at the end of a long multi-sentence paragraph.
9. THEME MODES: Standard, Professional/Work, and Playful sets.
10. DETERMINISTIC HASH SELECTION: Produces consistent emojis for identical inputs across runs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EmojiTheme(str, Enum):
    STANDARD = "standard"
    WORK = "work"
    PLAYFUL = "playful"


class EmojiDensity(str, Enum):
    SUBTLE = "subtle"        # Confidence >= 0.82
    BALANCED = "balanced"    # Confidence >= 0.70 (default)
    EXPRESSIVE = "expressive" # Confidence >= 0.55


# ─── Curated Emoji Sets by Category and Theme ─────────────────────────────────

_THEMED_EMOJI_SETS: dict[str, dict[str, list[str]]] = {
    EmojiTheme.STANDARD: {
        "celebratory": ["🏆", "🎉", "🥳", "🍾", "🥇"],
        "excited": ["🔥", "🚀", "⚡", "🤩"],
        "happy": ["😊", "✨", "🙌", "😄", "☀️"],
        "grateful": ["🙏", "💖", "🌸", "🤲"],
        "relieved": ["😌", "😮‍💨", "💆", "☕"],
        "humorous": ["😂", "🤣", "💀", "😆"],
        "supportive": ["💪", "🌟", "👊", "👏"],
        "thoughtful": ["🤔", "💡", "💭", "🧐"],
        "agreement": ["👍", "🤝", "✅", "🎯"],
        "cool": ["😎", "👌", "💯", "🪄"],
        "sad": ["😔", "🥺", "🌧️", "💔"],
        "frustrated": ["😤", "🤦", "💥", "🤯"],
        "apologetic": ["🙇", "🥺", "🙏"],
        "urgent": ["⚠️", "🚨", "⏳", "⏰"],
    },
    EmojiTheme.WORK: {
        "celebratory": ["🎉", "👏", "🏆"],
        "excited": ["🚀", "✨", "📈"],
        "happy": ["😊", "🙌", "✨"],
        "grateful": ["🙏", "🤝", "🌸"],
        "relieved": ["😌", "☕", "✅"],
        "humorous": ["😄", "🙂"],
        "supportive": ["👏", "💪", "🌟"],
        "thoughtful": ["💡", "🤔", "📊"],
        "agreement": ["👍", "🤝", "✅"],
        "cool": ["👌", "✅", "💡"],
        "sad": ["😔", "📉"],
        "frustrated": ["🤦", "⚠️"],
        "apologetic": ["🙇", "🙏"],
        "urgent": ["⚠️", "⏳", "🚨"],
    },
    EmojiTheme.PLAYFUL: {
        "celebratory": ["🎉", "🥳", "🍾", "🏆", "💃"],
        "excited": ["🔥", "🚀", "⚡", "🤯", "🤩"],
        "happy": ["😄", "✨", "🥳", "🕺", "🙌"],
        "grateful": ["💖", "🙏", "🥹", "💐"],
        "relieved": ["😮‍💨", "😌", "🥂", "🏖️"],
        "humorous": ["💀", "🤣", "😂", "😭"],
        "supportive": ["💪", "🔥", "👊", "🌟"],
        "thoughtful": ["🧐", "💡", "🧠", "💭"],
        "agreement": ["💯", "👌", "🤝", "🎯"],
        "cool": ["😎", "🤙", "🔥", "🪄"],
        "sad": ["🥺", "💔", "🌧️", "😭"],
        "frustrated": ["🤬", "💥", "😤", "🤦‍♂️"],
        "apologetic": ["🥺", "🙇‍♂️", "🙈"],
        "urgent": ["🚨", "🔥", "🏃💨", "⏰"],
    },
}

# Emotion Valence Mapping
_EMOTION_VALENCE: dict[str, str] = {
    "celebratory": "positive",
    "excited": "positive",
    "happy": "positive",
    "grateful": "positive",
    "relieved": "positive",
    "humorous": "positive",
    "supportive": "positive",
    "thoughtful": "neutral",
    "agreement": "positive",
    "cool": "positive",
    "sad": "negative",
    "frustrated": "negative",
    "apologetic": "neutral",
    "urgent": "negative",
}

# Intensity Tier Emojis (High-arousal vs. Mild)
_HIGH_INTENSITY_OVERRIDE: dict[str, list[str]] = {
    "celebratory": ["🏆", "🎉", "🍾"],
    "excited": ["🔥", "🚀", "⚡"],
    "humorous": ["🤣", "💀"],
    "frustrated": ["💥", "🤯"],
    "grateful": ["💖", "🙏"],
    "urgent": ["🚨", "⚠️"],
}


# ─── Multi-Lingual Emotion Lexicons (English + Roman Urdu + Urdu Script) ───────

_EMOTION_PATTERNS: dict[str, list[re.Pattern]] = {
    "celebratory": [
        # English
        re.compile(r"\b(?:we\s+won|crushed\s+it|killed\s+it|nailed\s+it|huge\s+win|milestone\s+reached|congrats|congratulations|proud\s+of\s+you|victory|champion|passed\s+the\s+test|celebrate|unlocked)\b", re.IGNORECASE),
        # Roman Urdu / Pakistani Slang
        re.compile(r"\b(?:phaar\s+diya|phod\s+diya|chha\s+gaye|scenes\s+on\s+hain|level\s+hai|kamaal\s+kar\s+diya|jeet\s+gaye|mubarak\s+ho|mubarakbaad|bohot\s+mubarak|shabash)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:جیت\s+گئے|مبارک|کامیابی|شاباش|فتح|مبارکباد)[\u0600-\u06FF]*"),
    ],
    "excited": [
        # English
        re.compile(r"\b(?:excited|thrilled|cannot\s+wait|cant\s+wait|lets\s+go|hyped|awesome|unbelievable|super\s+pumped|fire|insane|mind\s+blowing|launching|shipped|huge\s+milestone|breakthrough)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:zabardast|kamaal|maza\s+aagaya|phod\s+diya|lajawab|kya\s+baat\s+hai|dhamaka|bhaari|fit\s+hai)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:زبردست|کمال|لاجواب|کیا\s+بات\s+ہے|دھماکہ)[\u0600-\u06FF]*"),
    ],
    "happy": [
        # English
        re.compile(r"\b(?:happy|glad|delighted|pleased|wonderful|great\s+news|good\s+job|well\s+done|lovely|cheerful|blessed\s+day|great\s+to\s+see)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:bohot\s+khoob|achhi\s+baat|khushi\s+hui|dil\s+khush|bohot\s+pyara|bhai\s+bhai|chalo\s+acha\s+hua)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:خوشی|بہت\s+خوب|اچھی\s+بات|دل\s+خوش)[\u0600-\u06FF]*"),
    ],
    "grateful": [
        # English
        re.compile(r"\b(?:thank\s+you\s+so\s+much|grateful|appreciate\s+you|appreciate\s+it|thanks\s+a\s+lot|so\s+kind\s+of\s+you|much\s+obliged|blessed|means\s+a\s+lot)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:shukriya|bohot\s+shukriya|meherbani|jazakallah|dil\s+se\s+shukriya|nawazish)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:شکریہ|مہربانی|جزاک\s*اللہ|نوازش|دل\s+سے\s+شکریہ)[\u0600-\u06FF]*"),
    ],
    "relieved": [
        # English
        re.compile(r"\b(?:phew|finally|thank\s+goodness|what\s+a\s+relief|glad\s+that\s+is\s+over|glad\s+thats\s+over|safe\s+and\s+sound|all\s+done\s+finally)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:shukar\s+hai|jaan\s+chooti|chalo\s+bach\s+gaye|sukoon\s+mila|ab\s+chain\s+aya)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:شکر\s+ہے|جان\s+چھوٹی|سکون|الحمد\s*للہ)[\u0600-\u06FF]*"),
    ],
    "humorous": [
        # English
        re.compile(r"\b(?:hilarious|too\s+funny|cracking\s+up|dying\s+laughing|lmao|lol|rofl|made\s+me\s+laugh|joke|hysterical)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:hansi\s+aagayi|mazaaq|hansi\s+nahi\s+ruk\s+rahi|kitna\s+funny|chawwal|rofl)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:ہاہاہا|مذاق|ہنسی)[\u0600-\u06FF]*"),
    ],
    "supportive": [
        # English
        re.compile(r"\b(?:you\s+got\s+this|keep\s+it\s+up|stay\s+strong|proud\s+of\s+you|believe\s+in\s+you|cheering\s+for\s+you|dont\s+give\s+up|way\s+to\s+go)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:himmat\s+rakho|lage\s+raho|tension\s+na\s+lo|fiamanillah|koi\s+masla\s+nahi|hum\s+tumhare\s+saath\s+hain|jiyo)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:ہمت\s+رکھو|لگے\s+رہو|ٹینشن\s+نہ\s+لو)[\u0600-\u06FF]*"),
    ],
    "thoughtful": [
        # English
        re.compile(r"\b(?:what\s+if|thinking\s+about|let\s+me\s+ponder|good\s+point|interesting\s+idea|food\s+for\s+thought|curious\s+to\s+know|wondering|deep\s+thought)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:soch\s+raha\s+hoon|sochne\s+ki\s+baat\s+hai|ghor\s+karo|ajeeb\s+soch|kya\s+khayal\s+hai)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:سوچ|غور\s+کرو|کیا\s+خیال\s+ہے)[\u0600-\u06FF]*"),
    ],
    "agreement": [
        # English
        re.compile(r"\b(?:sounds\s+good|sounds\s+like\s+a\s+plan|deal|agreed|makes\s+sense|count\s+me\s+in|fair\s+enough|perfect\s+plan|absolutely\s+agree|spot\s+on)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:bilkul\s+sahi|pakka|theek\s+hai\s+done|done\s+karo|sahi\s+baat\s+hai|manzoor\s+hai|deal\s+done)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:بالکل\s+صحیح|اتفاق|منظور|پکا)[\u0600-\u06FF]*"),
    ],
    "cool": [
        # English
        re.compile(r"\b(?:cool|smooth|neat|clever|genius|brilliant|classy|solid\s+work|top\s+notch|flawless)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:fit\s+scene|solid\s+hai|class\s+hai|top\s+hai|bamb\s+hai|kamaal\s+cheez)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:کمال|شاندار|بہترین)[\u0600-\u06FF]*"),
    ],
    "sad": [
        # English
        re.compile(r"\b(?:sad|depressed|unfortunate|bummer|bad\s+news|feeling\s+down|crying|heartbroken|disappointed|terrible\s+outcome|miss\s+you)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:afsos|dukh\s+hua|bura\s+hua|udas|dil\s+toot\s+gaya|gham\s+hai|rona\s+aagaya)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:افسوس|دکھ|اداس|غم)[\u0600-\u06FF]*"),
    ],
    "frustrated": [
        # English
        re.compile(r"\b(?:frustrated|annoyed|irritating|hate\s+this|terrible|horrible|waste\s+of\s+time|sick\s+of\s+this|makes\s+no\s+sense|ridiculous|infuriating)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:dimag\s+kharab|ajeeb\s+masla|tang\s+aagaya|musibat|dimag\s+ki\s+dahi|pagal\s+hogaya|bakwas|gussa\s+aaraha)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:غصہ|مصیبت|بکواس|پریشان|تنگ\s+آ\s+گیا)[\u0600-\u06FF]*"),
    ],
    "apologetic": [
        # English
        re.compile(r"\b(?:my\s+bad|so\s+sorry|deeply\s+apologize|apologies|forgive\s+me|my\s+fault|my\s+mistake|pardon\s+me)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:maaf\s+karna|meri\s+galti|maaf\s+kijiye|sharminda\s+hoon|galti\s+ho\s+gayi)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:معاف\s+کرنا|معافی|میری\s+غلطی|شرمندہ)[\u0600-\u06FF]*"),
    ],
    "urgent": [
        # English
        re.compile(r"\b(?:asap|urgent|emergency|critical\s+bug|hotfix\s+now|immediately|right\s+now|action\s+required|deadline\s+is\s+today)\b", re.IGNORECASE),
        # Roman Urdu
        re.compile(r"\b(?:jaldi\s+karo|foran|abhi\s+ke\s+abhi|lazmi|urgent\s+hai|emergency\s+hai)\b", re.IGNORECASE),
        # Urdu Script
        re.compile(r"[\u0600-\u06FF]*(?:جلدی|فوراً|ہنگامی|ضروری)[\u0600-\u06FF]*"),
    ],
}


# ─── Negation Patterns & Sentiment Inverters ──────────────────────────────────

_NEGATION_CUES = re.compile(
    r"\b(?:not|never|hardly|scarcely|barely|no|doesnt|doesn't|dont|don't|wont|won't|isnt|isn't|arent|aren't|cannot|cant|can't|without|neither|nahi|nahin|na|mat|hargiz\s+nahi|نہیں|نہ|مت)\b",
    re.IGNORECASE,
)


# ─── Suppression Rules (Serious Contexts & Technical Dictation) ───────────────

# Serious / Somber contexts where inserting playful emojis would be insensitive/disrespectful
_SOMBER_SUPPRESSORS = re.compile(
    r"\b(?:condolence|condolences|passed\s+away|funeral|grief|mourning|died|death|fatal|suicide|hospitalized|icu|surgery|cancer|chemo|terminal|tragedy|accident|laid\s+off|layoffs|bankrupt|bankruptcy|foreclosure|lawsuit|arrested|police|janaza|wafat|inteqal|mar\s+gaye|mayyat|taziya|bimar|hospitaal|جنازہ|وفات|انتقال|تعزیت|ہسپتال|موت)\b",
    re.IGNORECASE,
)

# Developer / Technical / Factual commands
_TECHNICAL_SUPPRESSORS = re.compile(
    r"\b(?:api|endpoint|redis|database|server|docker|git|sql|commit|variable|function|deploy|ci|cd|migration|schema|bug\s+fix|unit\s+test|debug|pull\s+request|pr|branch|merge|kubernetes|k8s|oauth|jwt|ssl|tls|stdout|stderr|nginx|linux|bash)\b",
    re.IGNORECASE,
)

# Strong emotional overrides that still allow emojis even in technical text (e.g. "We shipped the API, let's go!")
_STRONG_TECHNICAL_EXCEPTIONS = re.compile(
    r"\b(?:awesome|lets\s+go|crushed\s+it|shipped|huge\s+milestone|breakthrough|zabardast|phaar\s+diya)\b",
    re.IGNORECASE,
)


# ─── Result Data Model ─────────────────────────────────────────────────────────

@dataclass
class EmotionDetectionResult:
    detected_emotion: str | None
    confidence: float
    emoji_inserted: str | None
    final_text: str
    valence: str = "neutral"          # "positive", "negative", "neutral"
    intensity: str = "medium"         # "low", "medium", "high"
    negation_detected: bool = False
    suppression_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_emotion": self.detected_emotion,
            "confidence": round(self.confidence, 3),
            "emoji_inserted": self.emoji_inserted,
            "final_text": self.final_text,
            "valence": self.valence,
            "intensity": self.intensity,
            "negation_detected": self.negation_detected,
            "suppression_reason": self.suppression_reason,
        }


# ─── Core Service Engine ──────────────────────────────────────────────────────

class EmojiService:
    """Context-aware, multi-lingual emotion detection and emoji insertion engine."""

    def __init__(
        self,
        default_theme: EmojiTheme = EmojiTheme.STANDARD,
        default_density: EmojiDensity = EmojiDensity.BALANCED,
    ):
        self.default_theme = default_theme
        self.default_density = default_density

    def process(
        self,
        text: str,
        enabled: bool = False,
        language: str = "en",
        theme: EmojiTheme | str | None = None,
        density: EmojiDensity | str | None = None,
    ) -> EmotionDetectionResult:
        """Process finalized text for contextual emoji insertion.

        Args:
            text: Punctuated, cleaned text to process.
            enabled: Master toggle. If False, returns text untouched immediately.
            language: Detected language code ('en', 'ur', 'ur_roman', etc.).
            theme: Emoji style ('standard', 'work', 'playful').
            density: Confidence threshold ('subtle', 'balanced', 'expressive').
        """
        # Guardrail 1: Strictly pass-through when disabled or input is empty
        if not enabled or not text or not text.strip():
            return EmotionDetectionResult(
                detected_emotion=None,
                confidence=0.0,
                emoji_inserted=None,
                final_text=text,
                valence="neutral",
                intensity="low",
                negation_detected=False,
                suppression_reason="disabled_or_empty",
            )

        active_theme = EmojiTheme(theme) if theme in EmojiTheme.__members__.values() else self.default_theme
        min_conf = self._get_density_threshold(density or self.default_density)

        # Guardrail 2: Serious/somber suppression (funerals, medical emergencies, etc.)
        if _SOMBER_SUPPRESSORS.search(text):
            return EmotionDetectionResult(
                detected_emotion=None,
                confidence=0.0,
                emoji_inserted=None,
                final_text=text,
                valence="neutral",
                intensity="low",
                negation_detected=False,
                suppression_reason="somber_context_suppressed",
            )

        # Split into sentences preserving punctuation
        sentence_chunks = re.split(r"(?<=[.?!])\s+", text.strip())
        if not sentence_chunks:
            sentence_chunks = [text.strip()]

        processed_sentences: list[str] = []
        overall_emotion: str | None = None
        highest_confidence: float = 0.0
        applied_emoji: str | None = None
        dominant_valence: str = "neutral"
        dominant_intensity: str = "medium"
        any_negation: bool = False
        suppression_note: str | None = None

        # Process each sentence; limit insertion to 1 emoji per utterance for tasteful elegance
        emoji_already_inserted = False

        for sentence in sentence_chunks:
            s_clean = sentence.strip()
            if not s_clean:
                continue

            # Analyze emotional tone for this sentence
            eval_res = self._analyze_sentence(s_clean, min_conf)

            if eval_res["suppressed"]:
                suppression_note = eval_res.get("suppression_reason", "technical_instruction")

            if eval_res["negation"]:
                any_negation = True

            emotion = eval_res["emotion"]
            conf = eval_res["confidence"]

            if emotion and conf >= min_conf:
                if conf > highest_confidence:
                    highest_confidence = conf
                    overall_emotion = emotion
                    dominant_valence = eval_res["valence"]
                    dominant_intensity = eval_res["intensity"]

                # Insert emoji if we haven't already inserted one in this utterance
                if not emoji_already_inserted and not eval_res["suppressed"]:
                    selected_emoji = self._select_emoji(
                        emotion=emotion,
                        theme=active_theme,
                        intensity=eval_res["intensity"],
                        seed_text=s_clean,
                    )
                    s_clean = self._insert_at_terminal(s_clean, selected_emoji)
                    applied_emoji = selected_emoji
                    emoji_already_inserted = True

            processed_sentences.append(s_clean)

        return EmotionDetectionResult(
            detected_emotion=overall_emotion,
            confidence=highest_confidence,
            emoji_inserted=applied_emoji,
            final_text=" ".join(processed_sentences),
            valence=dominant_valence,
            intensity=dominant_intensity,
            negation_detected=any_negation,
            suppression_reason=suppression_note if not applied_emoji else None,
        )

    # ─── Sentence Analysis & Scoring ──────────────────────────────────────────

    def _analyze_sentence(self, sentence: str, min_conf: float) -> dict[str, Any]:
        """Analyze a single sentence for emotions, negations, intensity, and suppression."""
        # Check technical suppression
        is_technical = bool(_TECHNICAL_SUPPRESSORS.search(sentence))
        has_strong_exception = bool(_STRONG_TECHNICAL_EXCEPTIONS.search(sentence))

        if is_technical and not has_strong_exception:
            return {
                "emotion": None,
                "confidence": 0.0,
                "suppressed": True,
                "suppression_reason": "technical_suppression",
                "negation": False,
                "valence": "neutral",
                "intensity": "low",
            }

        # Measure emotional intensity
        intensity = self._estimate_intensity(sentence)

        best_emotion: str | None = None
        highest_score: float = 0.0
        negation_triggered = False

        for emotion, patterns in _EMOTION_PATTERNS.items():
            for pat in patterns:
                for match in pat.finditer(sentence):
                    start_idx = match.start()
                    # Check for preceding negation in a 4-word window
                    prefix = sentence[max(0, start_idx - 35):start_idx]
                    is_negated = bool(_NEGATION_CUES.search(prefix))

                    if is_negated:
                        negation_triggered = True
                        # If a positive emotion is negated (e.g. "not happy"), invert or suppress
                        if _EMOTION_VALENCE.get(emotion) == "positive":
                            continue  # Suppress false positive

                    # Base confidence for matching a vetted cue
                    score = 0.75
                    if intensity == "high":
                        score += 0.15
                    elif intensity == "medium":
                        score += 0.05

                    if score > highest_score:
                        highest_score = min(0.98, score)
                        best_emotion = emotion

        # If a positive emotion was negated, assign mild disappointment if no other emotion caught
        if negation_triggered and not best_emotion:
            best_emotion = "frustrated"
            highest_score = 0.72

        valence = _EMOTION_VALENCE.get(best_emotion, "neutral") if best_emotion else "neutral"

        return {
            "emotion": best_emotion,
            "confidence": highest_score,
            "suppressed": False,
            "suppression_reason": None,
            "negation": negation_triggered,
            "valence": valence,
            "intensity": intensity,
        }

    # ─── Intensity & Arousal Calculation ──────────────────────────────────────

    @staticmethod
    def _estimate_intensity(sentence: str) -> str:
        """Estimate emotional arousal and intensity from linguistic cues."""
        score = 0
        # Exclamation marks
        if "!" in sentence:
            score += 1
        if re.search(r"!{2,}|\?!", sentence):
            score += 2

        # ALL-CAPS words of 3+ letters (e.g. "SO EXCITED", "ZABARDAST")
        caps_words = re.findall(r"\b[A-Z]{3,}\b", sentence)
        if len(caps_words) >= 1:
            score += 1

        # High-intensity adverbs (English + Roman Urdu)
        if re.search(r"\b(?:super|insanely|extremely|absolutely|completely|bohot|bohat|intehai|nihayat|bhai\s+sahab)\b", sentence, re.IGNORECASE):
            score += 1

        if score >= 2:
            return "high"
        elif score == 1:
            return "medium"
        return "low"

    # ─── Contextual Emoji Selection ───────────────────────────────────────────

    def _select_emoji(
        self,
        emotion: str,
        theme: EmojiTheme,
        intensity: str,
        seed_text: str,
    ) -> str:
        """Deterministically choose an appropriate emoji using a stable hash of the sentence."""
        theme_dict = _THEMED_EMOJI_SETS.get(theme, _THEMED_EMOJI_SETS[EmojiTheme.STANDARD])

        # If high intensity and available in override table, prefer high-arousal candidates
        candidates = theme_dict.get(emotion, ["✨"])
        if intensity == "high" and emotion in _HIGH_INTENSITY_OVERRIDE:
            override_pool = _HIGH_INTENSITY_OVERRIDE[emotion]
            # Intersect with theme candidates or use override
            candidates = [e for e in override_pool if e in candidates] or candidates

        # Use MD5 hash of seed_text to pick index deterministically
        hash_val = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16)
        idx = hash_val % len(candidates)
        return candidates[idx]

    # ─── Syntactic Placement ──────────────────────────────────────────────────

    @staticmethod
    def _insert_at_terminal(sentence: str, emoji: str) -> str:
        """Attach emoji cleanly at the terminal punctuation boundary."""
        sentence = sentence.strip()
        if not sentence:
            return emoji

        # Check for closing quote or parenthesis
        closing_chars = (')', '"', "'", '”', '’')
        trailing_wrapper = ""
        while sentence and sentence[-1] in closing_chars:
            trailing_wrapper = sentence[-1] + trailing_wrapper
            sentence = sentence[:-1].strip()

        # Handle terminal punctuation: ., !, ?
        if sentence.endswith((".", "!", "?", "…")):
            punct = sentence[-1]
            body = sentence[:-1].strip()
            return f"{body}{punct}{trailing_wrapper} {emoji}"

        return f"{sentence}{trailing_wrapper} {emoji}"

    @staticmethod
    def _get_density_threshold(density: EmojiDensity | str) -> float:
        if density == EmojiDensity.SUBTLE or density == "subtle":
            return 0.82
        elif density == EmojiDensity.EXPRESSIVE or density == "expressive":
            return 0.55
        return 0.70  # Balanced default


# Global singleton instance
emoji_service = EmojiService()

