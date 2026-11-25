""
import decimal

D = decimal.Decimal
CTX = decimal.Context(prec=80)
ROUNDING = {"upper": decimal.ROUND_CEILING, "lower": decimal.ROUND_FLOOR, "point": decimal.ROUND_HALF_EVEN}
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def part(value, role, digits=0, *, scale=1, thousands=False, source=None):
    ""
    out = {"value": value, "role": role, "digits": digits}
    if scale != 1:
        out["scale"] = scale
    if thousands:
        out["thousands"] = True
    if role == "count":
        if not isinstance(value, int) or isinstance(value, bool) or digits != 0:
            raise ValueError(f"a count is a whole int printed at 0 digits: {value!r}, {digits}")
        n = value * scale
        out.update(printed=f"{n:,}" if thousands else str(n), printed_value=n)
    else:
        if role not in ROUNDING:
            raise ValueError(f"unknown role {role!r}")
        x = CTX.multiply(D(value), D(scale))
        r = x.quantize(D(1).scaleb(-digits), rounding=ROUNDING[role], context=CTX)
        out.update(printed=f"{r:,f}" if thousands else f"{r:f}",
                   printed_value=float(CTX.divide(r, D(scale))))
    if source is not None:
        out["source"] = source
    return out


def display(template, **parts):
    ""
    return {"template": template, "parts": parts,
            "text": template.format(**{n: p["printed"] for n, p in parts.items()})}


def word(n, initial=False):
    ""
    w = WORDS[n]
    return w[0].upper() + w[1:] if initial else w
