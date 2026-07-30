"""
Generates tests/fixtures/sample.warc.wet.gz: a small, synthetic, WET-format
WARC file used by CI so the pipeline can be exercised end to end without any
network access to Common Crawl. All text below is original/synthetic (no
copyrighted material) and deliberately includes documents that should survive
ingest_clean.py's filters, some that shouldn't (too short, non-English), and
one intentional near-duplicate pair so dedup_cluster.py's MinHash stage has
something real to remove.

Run this to regenerate the fixture (only needed if you change the documents):
    python tests/fixtures/generate_fixture.py
"""

from io import BytesIO
from pathlib import Path

from warcio.warcwriter import WARCWriter

OUT_PATH = Path(__file__).resolve().parent / "sample.warc.wet.gz"

GOOD_DOCS = [
    (
        "http://example-news.test/local-council-budget",
        "The town council met on Tuesday evening to review the proposed budget for the "
        "coming fiscal year. Several residents spoke during the public comment period, "
        "raising concerns about road maintenance funding and the proposed increase to the "
        "parks department allocation. The council chair noted that revenue projections had "
        "improved slightly since the last quarterly review, largely due to higher than "
        "expected sales tax receipts. A final vote on the budget is scheduled for the next "
        "regular meeting in three weeks. Residents can submit written comments to the clerk's "
        "office until then. The full budget document is available for review at the public "
        "library and on the town website.",
    ),
    (
        "http://example-blog.test/growing-tomatoes",
        "Growing tomatoes in a home garden rewards patience more than almost any other "
        "vegetable. Start seedlings indoors about six weeks before the last expected frost, "
        "keeping them under a bright light source so they don't grow leggy. Once the soil has "
        "warmed and the danger of frost has passed, harden off the young plants gradually "
        "before transplanting them outside. Consistent watering matters more than fertilizer "
        "for the first few weeks, since inconsistent moisture is the most common cause of "
        "blossom end rot later in the season. Staking or caging the plants early prevents "
        "stem damage once the fruit starts to weigh down the branches. Most varieties are "
        "ready to harvest roughly two months after transplanting.",
    ),
    (
        "http://example-sports.test/weekend-recap",
        "The home team came from behind twice in Saturday's match, eventually settling for a "
        "draw after a tense final ten minutes. The visiting side opened the scoring early in "
        "the second half, catching the defense out of position on a quick counterattack. "
        "A late equalizer came from a corner kick that deflected off two defenders before "
        "crossing the line, much to the relief of the home crowd. The manager praised the "
        "team's resilience afterward but acknowledged that defensive organization needs work "
        "before next week's away fixture. Attendance was reported at just over eleven "
        "thousand, a season high for a midweek game.",
    ),
    (
        "http://example-tech.test/battery-life-tips",
        "Extending the battery life of a laptop usually comes down to a handful of habits "
        "rather than any single setting. Lowering screen brightness has the single biggest "
        "impact for most users, followed by closing background applications that keep polling "
        "the network. Many laptops also ship with a battery health mode that limits the "
        "maximum charge to around eighty percent, which can meaningfully slow long-term "
        "capacity loss if the machine is usually plugged in. Avoiding extreme temperatures, "
        "both hot and cold, also helps preserve the battery over a few years of daily use. "
        "None of these changes will double battery life on their own, but together they add "
        "up to a noticeably longer runtime between charges.",
    ),
    (
        "http://example-recipes.test/weeknight-soup",
        "This weeknight soup comes together in under thirty minutes and uses mostly pantry "
        "staples. Start by softening a diced onion and two cloves of garlic in a large pot "
        "with a little olive oil over medium heat. Add a can of diced tomatoes, a quart of "
        "vegetable broth, and a cup of small pasta, then simmer until the pasta is just "
        "tender. Stir in a couple of handfuls of chopped greens at the very end so they wilt "
        "without overcooking. Season generously with salt, pepper, and a splash of vinegar to "
        "brighten the flavor before serving. Leftovers keep well in the refrigerator for a "
        "few days, though the pasta will continue to absorb liquid overnight.",
    ),
    (
        "http://example-travel.test/train-vs-plane",
        "Choosing between a train and a short-haul flight usually comes down to more than "
        "just ticket price. Trains typically drop travelers directly into city centers, "
        "avoiding the additional time and cost of getting to and from an airport located well "
        "outside town. Security lines are also generally shorter and faster for rail travel, "
        "which can offset a longer total journey time compared to flying. On routes under "
        "roughly four hours, door-to-door travel time is often similar or even faster by "
        "train once airport transfers and security are factored in. For longer distances, "
        "flying usually wins on total time, though at a higher environmental cost per "
        "passenger.",
    ),
]

# near-duplicate of the first good doc (minor edits) -- MinHash dedup should catch this pair
NEAR_DUP_DOC = (
    "http://example-news-mirror.test/local-council-budget-copy",
    "The town council met on Tuesday evening to review the proposed budget for the "
    "upcoming fiscal year. Several residents spoke during the public comment period, "
    "raising concerns about road maintenance funding and the proposed increase to the "
    "parks department budget. The council chair noted that revenue projections had "
    "improved slightly since the last quarterly review, largely due to higher than "
    "expected sales tax receipts. A final vote on the budget is scheduled for the next "
    "regular meeting in three weeks. Residents can submit written comments to the clerk's "
    "office until then. The full budget document is available for review at the public "
    "library and on the town website.",
)

SHORT_DOCS = [
    ("http://example-junk.test/stub-page", "Page not found. Try again later."),
    ("http://example-junk.test/under-construction", "This page is under construction. Check back soon."),
    ("http://example-junk.test/empty-listing", "No results. No results. No results."),
]

FOREIGN_DOC = (
    "http://example-news.test/fr/conseil-municipal",
    "Le conseil municipal s'est réuni mardi soir pour examiner le budget proposé pour "
    "l'année fiscale à venir. Plusieurs résidents ont pris la parole pendant la période de "
    "commentaires publics, exprimant leurs préoccupations concernant le financement de "
    "l'entretien des routes et l'augmentation proposée pour le département des parcs. Le "
    "président du conseil a noté que les prévisions de revenus s'étaient légèrement "
    "améliorées depuis le dernier examen trimestriel, principalement en raison de recettes "
    "fiscales sur les ventes plus élevées que prévu. Un vote final sur le budget est prévu "
    "lors de la prochaine réunion ordinaire dans trois semaines.",
)

REPETITIVE_DOC = (
    "http://example-spam.test/click-here",
    "Click here now! Click here now! Click here now! Best deals click here now! "
    "Click here now! Click here now! Limited time click here now! Click here now! "
    "Click here now! Click here now! Click here now! Click here now!",
)


def build_records():
    docs = list(GOOD_DOCS) + [NEAR_DUP_DOC] + SHORT_DOCS + [FOREIGN_DOC, REPETITIVE_DOC]
    return docs


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as raw_f:
        writer = WARCWriter(raw_f, gzip=True)  # per-record gzip members, matching real .warc.wet.gz files

        warcinfo = writer.create_warcinfo_record(str(OUT_PATH), info={"software": "training-data-refinery fixture"})
        writer.write_record(warcinfo)

        for uri, text in build_records():
            record = writer.create_warc_record(
                uri,
                "conversion",
                payload=BytesIO(text.encode("utf-8")),
                warc_content_type="text/plain",
                warc_headers_dict={"WARC-Date": "2024-01-01T00:00:00Z"},
            )
            writer.write_record(record)

    n_docs = len(build_records())
    print(f"Wrote {n_docs} synthetic records to {OUT_PATH}")


if __name__ == "__main__":
    main()
