"""Fail-closed G1 paper-reproduction preflight; no result runner exists at G0."""

from __future__ import annotations

from pathlib import Path


REQUIRED_ASSETS = (
    Path("configs/paper-reproduction.json"),
    Path("data/paper"),
    Path("src/sasto/paper_runner.py"),
)
REQUIRED_NOTICES = (
    Path("LICENSE_STATUS.md"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("DATA_NOTICE.md"),
)


def main() -> int:
    """Report the exact unresolved G1 gates and deliberately return nonzero."""
    missing = [str(path) for path in REQUIRED_ASSETS if not path.exists()]
    missing_notices = [str(path) for path in REQUIRED_NOTICES if not path.is_file()]
    license_status = Path("LICENSE_STATUS.md")
    unresolved_license = not license_status.is_file() or "Status: UNRESOLVED" not in license_status.read_text(
        encoding="utf-8"
    )
    gates = []
    if missing:
        gates.append("missing config/data/runner assets: {}".format(", ".join(missing)))
    if missing_notices:
        gates.append("missing required notices: {}".format(", ".join(missing_notices)))
    if unresolved_license:
        gates.append("license selection remains an external user gate")
    if not gates:
        gates.append("a real G1 paper runner has not been implemented or validated")
    print("G1 UNAVAILABLE: make reproduce-paper does not reproduce paper results.")
    print("Preflight failed closed: {}.".format("; ".join(gates)))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
