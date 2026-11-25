""

from __future__ import annotations

import json
import re
import sys

from screen_common import HERE, RULES_MD, sha256_file, write_lock


def main() -> int:
    rules_sha = sha256_file(RULES_MD)
    script = HERE / "run_headline.sh"
    text = script.read_text()
    new = re.sub(r"^RULES_SHA256=\S+", f"RULES_SHA256={rules_sha}", text, count=1, flags=re.M)
    if new != text:
        script.write_text(new)
    record = write_lock()
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
