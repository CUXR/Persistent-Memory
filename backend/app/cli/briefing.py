"""Print a person briefing by person ID (face path) or by name (name path)."""
import argparse
from uuid import UUID

DISPLAY_MODES = ("nothing", "name", "name_and_facts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-id", type=UUID, required=True)
    parser.add_argument("--max-facts", type=int, default=5, help="Facts per category (page size)")
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief")
    brief.add_argument("person_id", type=UUID, help="Person ID supplied by the face tracker")
    brief.add_argument("--mode", choices=DISPLAY_MODES,
                       help="Display mode; defaults to the owner's stored setting")
    by_name = sub.add_parser("by-name")
    by_name.add_argument("name")
    more = sub.add_parser("more", help="Next page of facts from a next_cursor")
    more.add_argument("cursor")
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
                result = service.brief(args.person_id, max_facts=args.max_facts, display_mode=args.mode)
            elif args.command == "more":
                result = service.more_facts(args.cursor, limit=args.max_facts)
            else:
                result = service.brief_by_name(args.name, max_facts=args.max_facts)
        except (PersonNotFoundError, InvalidCursorError, ValueError) as exc:
            raise SystemExit(str(exc))
        print(result.model_dump_json(indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
