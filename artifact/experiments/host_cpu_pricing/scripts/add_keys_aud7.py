""
import hashlib
import json
import re
import sys
from pathlib import Path

P = Path(sys.argv[1]).resolve()
U = "records/tiers/fidelity/ucl0h3"
HERE = Path(__file__).resolve().parent


def sha(rel):
    return hashlib.sha256((P / rel).read_bytes()).hexdigest()


def final_pin(name):
    for line in (P / U / "SHA256SUMS.final").read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() in (f"./{name}", name):
            return parts[0]
    sys.exit(f"add_keys_aud7.py: {U}/SHA256SUMS.final names no {name}: stopped")


code = f"{U}/cl0h_host_study.py"
ch = sha(code)
if ch != final_pin("cl0h_host_study.py") or not ch.startswith("e463b78c"):
    sys.exit(f"add_keys_aud7.py: {code} has sha256 {ch[:16]}, not the pin e463b78c of SHA256SUMS.final: stopped")
lines = (P / code).read_text().splitlines()
hits = [i for i, ln in enumerate(lines, 1) if re.fullmatch(r"CAPS_PER_TARGET = (\d+)", ln.strip())]
if len(hits) != 1:
    sys.exit(f"add_keys_aud7.py: {code} holds {len(hits)} lines 'CAPS_PER_TARGET = n', not one: stopped")
ln = hits[0]
rule = lines[ln - 2].strip()
if not rule.startswith("#: CL0's cap rule: this many log-spaced caps per target"):
    sys.exit(f"add_keys_aud7.py: {code}:{ln - 1} is not the cap rule's comment: stopped")
n = int(re.fullmatch(r"CAPS_PER_TARGET = (\d+)", lines[ln - 1].strip()).group(1))

keys = f"{U}/keys.json"
kh = sha(keys)
if kh != final_pin("keys.json") or not kh.startswith("6a257955"):
    sys.exit(f"add_keys_aud7.py: {keys} has sha256 {kh[:16]}, not the pin 6a257955: stopped")
kb = json.loads((P / keys).read_text())["keys"]["hostboundcellspub"]
disp, deriv, value = kb["display"], kb["derivation"], kb.get("value")
for t, where in ((disp, "display"), (deriv, "derivation")):
    if f"({n} log-spaced caps" not in t and f"{n} log-spaced caps per target" not in t:
        sys.exit(f"add_keys_aud7.py: keys.json hostboundcellspub/{where} does not name {n} log-spaced caps: stopped")
m = re.search(r"of (\d+) cap cells", disp)
if not m or int(m.group(1)) % n:
    sys.exit(f"add_keys_aud7.py: keys.json's cap cells are not a whole number of targets of {n} caps: stopped")

out = {"what": "Derived values of step aud7:assemble (the third fix round's assembly, 2026-11-20), by group (decl: "
               "declared settings and counts). Each value names its inputs (file relative to the paper folder, sha256, "
               "field or line, value) and its arithmetic. Written by add_keys_aud7.py beside this file; never edited "
               "by hand.",
       "decl": {"hostcapsper": dict(
           value=n,
           inputs=[dict(file=code, sha256=ch, field=f"line {ln}", value=lines[ln - 1].strip(), comment=rule),
                   dict(file=keys, sha256=kh, field="keys/hostboundcellspub/display", value=disp),
                   dict(file=keys, sha256=kh, field="keys/hostboundcellspub/derivation", value=deriv)],
           arithmetic=f"the integer n of 'CAPS_PER_TARGET = n' on line {ln} (the host study's cap rule: n log-spaced "
                      f"energy caps per target); keys.json's {m.group(1)} cap cells are {int(m.group(1)) // n} targets "
                      f"of {n} caps")}}
dst = HERE / "derived_values.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"hostcapsper = {n} (line {ln} of cl0h_host_study.py, e463b78c; keys.json 6a257955 agrees); "
      f"wrote {dst.name} {hashlib.sha256(dst.read_bytes()).hexdigest()[:8]}")
