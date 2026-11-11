"""Issue, list, and revoke bearer tokens for the HTTP API."""
import argparse
from uuid import UUID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Issue a token; it is printed once and never stored")
    create.add_argument("--owner-id", type=UUID, required=True)
    create.add_argument("--label", required=True, help='Which client holds it, e.g. "RayNeo glasses"')
    listing = sub.add_parser("list")
    listing.add_argument("--owner-id", type=UUID, required=True)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("token_id", type=UUID)
    args = parser.parse_args()
    # Allow --help without database settings or application model initialization.
    from ..core.database import make_engine, make_session_factory
    from ..crud.api_tokens import issue_token, list_tokens, revoke_token

    engine = make_engine()
    try:
        with make_session_factory(engine).begin() as session:
            if args.command == "create":
                try:
                    token, raw = issue_token(session, args.owner_id, args.label)
                except ValueError as exc:
                    raise SystemExit(str(exc))
                print(f"token_id: {token.id}\ntoken:    {raw}")
            elif args.command == "list":
                for token in list_tokens(session, args.owner_id):
                    state = f"revoked {token.revoked_at:%Y-%m-%d}" if token.revoked_at else "active"
                    print(f"{token.id}  {token.created_at:%Y-%m-%d}  {state:18}  {token.label}")
            elif not revoke_token(session, args.token_id):
                raise SystemExit(f"Token id={args.token_id} not found")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
