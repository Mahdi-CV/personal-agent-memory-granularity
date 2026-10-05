"""FAR interference probe: is the general assistant still intact?

WHY A SEPARATE SET

The NEAR probe (TOFU held-out authors 10-19) measures whether installing memory
makes the model fabricate the WRONG person's facts. It cannot tell us whether
general capability survived, because every held-out question is still a TOFU
author biography and shares the trained format.

The D2L paper's own Table 8 reports that with an unrelated query, D2L scores
BELOW the unmodified base (0.096/0.126 vs 0.201/0.175) and hypothesizes the
model "acquires a strong prior that subsequent queries always relate to the
internalized knowledge." For a persistent personal-agent adapter that is a
first-class cost, so it needs measuring directly rather than inferring.

DESIGN

Short, unambiguous, single-fact questions with a small closed set of acceptable
surface forms, so grading is exact-match-ish and needs no model judge. Topics
deliberately span domains (science, geography, history, math, language, code)
and deliberately avoid authors, books, awards and birthplaces, which are the
TOFU domain.

THE FILTER THAT MAKES THIS VALID

A question the base model gets wrong measures nothing about interference. So
`filter_answerable` runs the set against the unmodified base FIRST and keeps only
the items it answers correctly. Every arm is then scored on that same surviving
subset. A drop therefore means the memory mechanism destroyed a capability the
base demonstrably had, which is the only claim worth making.
"""
import re

# (question, [accepted answer substrings, lowercased])
FAR_ITEMS = [
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("What planet is known as the Red Planet?", ["mars"]),
    ("What gas do plants absorb from the atmosphere during photosynthesis?",
     ["carbon dioxide", "co2", "co²"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("How many sides does a hexagon have?", ["six", "6"]),
    ("What is the boiling point of water in degrees Celsius at sea level?",
     ["100", "one hundred"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci", "vinci"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    ("In which country are the pyramids of Giza located?", ["egypt"]),
    ("What is the chemical formula for water?", ["h2o", "h₂o", "h 2 o"]),
    ("How many minutes are in a full day?", ["1440", "1,440"]),
    ("What is the freezing point of water in degrees Fahrenheit?", ["32"]),
    ("Which language has the most native speakers in the world?",
     ["chinese", "mandarin"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What organ in the human body pumps blood?", ["heart"]),
    ("What is the tallest mountain above sea level on Earth?",
     ["everest"]),
    ("How many bits are in a byte?", ["8", "eight"]),
    ("What does CPU stand for in computing?", ["central processing unit"]),
    ("What is 15 percent of 200?", ["30", "thirty"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
    ("What is the currency of the United Kingdom?",
     ["pound", "sterling", "gbp"]),
    ("How many players are on a standard soccer team on the field?",
     ["11", "eleven"]),
    ("What is the hardest naturally occurring substance?", ["diamond"]),
    ("In what year did the Second World War end?", ["1945"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("What is the plural of the word 'child'?", ["children"]),
    ("Which element has the atomic number 1?", ["hydrogen"]),
    ("What is 7 multiplied by 8?", ["56", "fifty-six", "fifty six"]),
]


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def grade_far(gen, accepted):
    """Substring match on the normalized generation.

    Deliberately generous: a verbose but correct answer still counts, because
    the question is whether the KNOWLEDGE survived, not whether the model became
    more talkative. Numeric answers are matched as whole tokens so that "12"
    does not match inside "120".
    """
    g = norm(gen)
    gt = set(g.split())
    for a in accepted:
        an = norm(a)
        if not an:
            continue
        if an.replace(" ", "").isdigit():
            if an in gt:
                return 1
        elif an in g:
            return 1
    return 0


def filter_answerable(gen_fn):
    """Keep only items the unmodified base answers correctly.

    gen_fn(question) -> generated text. Returns (kept, dropped).
    """
    kept, dropped = [], []
    for q, acc in FAR_ITEMS:
        g = gen_fn(q)
        (kept if grade_far(g, acc) else dropped).append((q, acc, g))
    return kept, dropped
