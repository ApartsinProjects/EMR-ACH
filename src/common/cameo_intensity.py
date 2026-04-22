"""CAMEO-event-code → Peace / Tension / Violence intensity mapping.

The benchmark's GDELT-CAMEO target was reframed (2026-04-22) from the 4-class
QuadClass (VC / MC / VK / MK) to a 3-class **intensity** taxonomy:

  Peace    — Roots 01-09 (cooperative: statements, appeals, intents,
                        consultations, diplomatic/material cooperation,
                        aid, yielding, investigations)
  Tension  — Roots 10-17 (non-violent friction: demands, disapproval,
                        rejection, threats, protests, force postures,
                        reduced relations, coercion incl. sanctions)
  Violence — Roots 18-20 (physical force: assault, fight, unconventional
                        mass violence)

The intensity reduction is ordinal (Peace < Tension < Violence), which enables
ordinal-error metrics in addition to nominal accuracy.

The old quad-class mapping is preserved in `quad_to_label` for comparability
with MIRAI-era numbers.
"""
from __future__ import annotations

INTENSITY_CLASSES = ("Peace", "Tension", "Violence")

# Ordered for ordinal comparison: index differences measure "intensity distance".
INTENSITY_RANK: dict[str, int] = {"Peace": 0, "Tension": 1, "Violence": 2}

# Legacy 4-class quad labels. Provided for backwards compat; prefer intensity.
QUAD_LABELS = ("Verbal Cooperation", "Material Cooperation", "Verbal Conflict", "Material Conflict")
QUAD_CODES = ("VC", "MC", "VK", "MK")


def root_code(event_base_code: str | int | None) -> int | None:
    """Extract the 2-digit CAMEO root from an EventBaseCode.

    EventBaseCode is a 3- or 4-digit string/int (e.g. '013' → root 01,
    '0141' → root 01, '1832' → root 18). Returns None if unparseable.
    """
    if event_base_code is None:
        return None
    s = str(event_base_code).strip()
    if not s:
        return None
    # Left-pad short codes (some files omit leading zero: "13" meaning "013")
    s = s.zfill(3)
    try:
        return int(s[:2])
    except ValueError:
        return None


def root_to_intensity(root: int | None) -> str | None:
    """Map CAMEO root code → Peace / Tension / Violence. None if root is invalid."""
    if root is None:
        return None
    if 1 <= root <= 9:
        return "Peace"
    if 10 <= root <= 17:
        return "Tension"
    if 18 <= root <= 20:
        return "Violence"
    return None


def event_to_intensity(event_base_code: str | int | None) -> str | None:
    """Convenience: EventBaseCode → intensity string.

        >>> event_to_intensity('013')   # root 01 = Make public statement
        'Peace'
        >>> event_to_intensity('1411')  # root 14 = Protest
        'Tension'
        >>> event_to_intensity('1920')  # root 19 = Fight
        'Violence'
    """
    return root_to_intensity(root_code(event_base_code))


def quad_to_intensity_maybe(quad: int | str | None) -> str | None:
    """BEST-EFFORT mapping from QuadClass (1..4) to intensity.

    Lossy: QuadClass 4 (Material Conflict) spans both Tension (roots 14-17:
    protest, force posture, reduce relations, coerce) and Violence (roots
    18-20: assault, fight, mass violence). We can't disambiguate without the
    root code, so Material Conflict defaults to "Tension" (more common by
    base rate).

    Prefer `event_to_intensity(event_base_code)` when the root is available.
    """
    if quad in (1, "1", "VC"):
        return "Peace"
    if quad in (2, "2", "MC"):
        return "Peace"
    if quad in (3, "3", "VK"):
        return "Tension"
    if quad in (4, "4", "MK"):
        return "Tension"   # see docstring — lossy fallback
    return None


def quad_to_label(quad: int | str | None) -> str | None:
    """Legacy: QuadClass (1-4) → full class label ('Verbal Cooperation' etc.)."""
    if quad in (1, "1", "VC"):
        return "Verbal Cooperation"
    if quad in (2, "2", "MC"):
        return "Material Cooperation"
    if quad in (3, "3", "VK"):
        return "Verbal Conflict"
    if quad in (4, "4", "MK"):
        return "Material Conflict"
    return None


# Sample definitions used in the hypothesis definitions block of every FD.
INTENSITY_DEFINITIONS: dict[str, str] = {
    "Peace":    ("Cooperative or neutral interaction between the two parties on the target date: "
                 "public statements, diplomatic meetings, trade/aid, agreements, or factual consultations. "
                 "Maps to CAMEO root codes 01-09."),
    "Tension":  ("Non-violent friction: demands, disapproval, rejection of proposals, verbal threats, "
                 "protests, mobilization/force posture, reduced relations, sanctions, or other coercion "
                 "short of physical force. Maps to CAMEO root codes 10-17."),
    "Violence": ("Physical use of force: assault, armed clashes, military operations, or unconventional "
                 "mass violence. Maps to CAMEO root codes 18-20."),
}
