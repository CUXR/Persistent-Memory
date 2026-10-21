"""Print a person briefing by person ID (face path) or by name (name path)."""
import argparse
from uuid import UUID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", type=UUID, required=True)
    parser.add_argument("--max-facts", type=int, default=5, help="Facts per category")
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief")
    brief.add_argument("person_id", type=UUID, help="Person ID supplied by the face tracker")
    by_name = sub.add_parser("by-name")
    by_name.add_argument("name")
    args = parser.parse_args()
    if args.max_facts < 0:
        parser.error("--max-facts must be non-negative")
    # Allow --help without database settings or application model initialization.
    from ..crud.briefing_queries import PersonNotFoundError
    from ..crud.memory_store import MemoryStore
    from ..services.briefing import BriefingService

    store = MemoryStore(owner_user_id=args.owner_id)
    store.initialize(create_schema=False)
    try:
        service = BriefingService(store)
        if args.command == "brief":
            try:
                result = service.brief(args.person_id, max_facts=args.max_facts)
            except PersonNotFoundError as exc:
                raise SystemExit(str(exc))
        else:
            result = service.brief_by_name(args.name, max_facts=args.max_facts)
        print(result.model_dump_json(indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
