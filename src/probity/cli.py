"""``probity`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from probity.api.main import create_app


def _export_openapi(_: argparse.Namespace) -> int:
    print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probity")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export-openapi", help="Print the generated OpenAPI document").set_defaults(
        func=_export_openapi
    )
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
