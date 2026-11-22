"""Load the committed dataset into the eval database.

    python -m eval.retrieval.data.load_db [--reset]

Reads the committed JSONL, embeds the summaries with bge-m3 (or reads the
embeddings cached by an earlier run), creates the eval-owned tables where they
are missing, and upserts everything for the eval owner. The load can be
repeated. ``--reset`` first deletes what earlier loads left for that owner,
which is how rows the dataset no longer contains are removed. When
``queries.jsonl`` exists, its queries are copied into ``eval_queries`` in the
same transaction, for people who read the eval branch.

The load records the dataset's fingerprint (``db.dataset_fingerprint``) next
to the rows; a run over the pg store checks it (``db.check_corpus``) so that a
regenerated dataset that was never loaded cannot be benchmarked against the
old rows under the same ids.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .. import config, dataset, db
from ..local_models import BgeM3Embedder, load_or_compute_corpus_embeddings
from ..schema import Query


def _committed_queries() -> list[Query] | None:
    """The queries of ``queries.jsonl``, or None while that file has not been generated."""

    return dataset.load_queries() if config.QUERIES_PATH.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the load. Returns the exit code: 1 if the database then differs from the dataset."""

    parser = argparse.ArgumentParser(
        prog="python -m eval.retrieval.data.load_db",
        description="Load the committed benchmark dataset into the eval database.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the eval owner's existing rows before loading",
    )
    args = parser.parse_args(argv)

    # Checked before the model loads, so a missing EVAL_DATABASE_URL fails at once.
    database_url = db.resolve_url()
    people = dataset.load_people()
    conversations = dataset.load_conversations()
    queries = _committed_queries()
    print(f"Embedding {len(conversations)} summaries with {config.EMBEDDING_MODEL}...", flush=True)
    embeddings = load_or_compute_corpus_embeddings(conversations, BgeM3Embedder())

    print(f"Loading {len(people)} people and {len(conversations)} conversations...", flush=True)
    with db.connect(database_url) as conn:
        db.ensure_eval_tables(conn)
        # One transaction around both steps: a load that fails takes its reset back with it.
        with conn.transaction():
            if args.reset:
                db.reset_eval_data(conn)
            db.load_dataset(conn, people, conversations, embeddings)
            if queries is not None:
                db.load_queries(conn, queries)
        counts = db.count_rows(conn)
        stored = db.corpus_fingerprint(conn)

    expected = {
        "users": 1,
        "people": len(people),
        "episodes": len(conversations),
        "episode_participants": len(conversations),
        "eval_episode_dialogues": len(conversations),
        "eval_episode_embeddings": len(conversations),
    }
    fingerprint = db.dataset_fingerprint(people, conversations)
    print(f"Loaded the dataset for owner {config.OWNER_ID} ({config.OWNER_DISPLAY_NAME}).")
    print(f"Corpus fingerprint: {stored}")
    if queries is None:
        print("No queries.jsonl yet: eval_queries was left as it is.")
    else:
        print(f"Queries copied into eval_queries: {len(queries)}")
    print("Rows in the database for that owner:")
    for table in db.COUNTED_TABLES:
        print(f"  {table:<24}{counts[table]:>6}")
    differing = [table for table in db.COUNTED_TABLES if counts[table] != expected[table]]
    if differing:
        details = ", ".join(f"{table} (dataset: {expected[table]})" for table in differing)
        print(
            f"The database does not match the dataset in: {details}. "
            "Rows left by an earlier load are removed with --reset.",
            file=sys.stderr,
        )
    if stored != fingerprint:
        print(
            f"The database records the corpus fingerprint {stored}, the dataset's is {fingerprint}.",
            file=sys.stderr,
        )
    return 1 if differing or stored != fingerprint else 0


if __name__ == "__main__":
    raise SystemExit(main())
