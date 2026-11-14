"""Print a person briefing by person ID or by name, or a page of a person's facts."""
import argparse
from uuid import UUID

CATEGORIES = ("hobby", "affiliation", "visual_descriptor")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", type=UUID, required=True)
    parser.add_argument("--max-facts", type=int, default=5, help="Facts per category (page size)")
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief")
    brief.add_argument("person_id", type=UUID, help="Person ID, e.g. from the face tracker")
    by_name = sub.add_parser("by-name")
    by_name.add_argument("name")
    facts = sub.add_parser("facts", help="One page of a person's facts in one category")
    facts.add_argument("person_id", type=UUID)
    facts.add_argument("category", choices=CATEGORIES)
    facts.add_argument("--cursor", help="A next_cursor from the previous page")
    args = parser.parse_args()
    if args.max_facts < 0:
        parser.error("--max-facts must be non-negative")
    # Allow --help without database settings or application model initialization.
    from ..crud.briefing_queries import InvalidCursorError, PersonNotFoundError
    from ..crud.memory_store import MemoryStore
    from ..services.briefing import BriefingService

    store = MemoryStore(owner_user_id=args.owner_id)
    store.initialize(create_schema=False)
    try:
        service = BriefingService(store)
        try:
            if args.command == "brief":
                result = service.brief(args.person_id, max_facts=args.max_facts)
            elif args.command == "facts":
                result = service.facts(args.person_id, args.category, cursor=args.cursor,
                                       limit=args.max_facts)
            else:
                result = service.brief_by_name(args.name, max_facts=args.max_facts)
        except (PersonNotFoundError, InvalidCursorError, ValueError) as exc:
            raise SystemExit(str(exc))
        print(result.model_dump_json(indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
