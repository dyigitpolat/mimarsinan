""
WIDE, R3 = 131072, 8192
admitted = {256: [160, 192, 224, 256], 576: [216, 288, 360, 432, 504, 576]}
b2 = []
for budget, buf in ((WIDE, 0), (WIDE, 1), (R3, 0), (R3, 1)):
    for slots in (16, 8, 4, 2):
        for n in (256, 576):
            b2 += [f"E2 {n} {budget} {g} {buf} {slots}" for g in reversed(admitted[n])]
lines = ["W0 256", f"E2 256 {WIDE} 256 0 0", f"E2 576 {WIDE} 432 0 0"] + b2 + [f"E2 144 {WIDE} 144 0 16"]
print("\n".join(lines))
