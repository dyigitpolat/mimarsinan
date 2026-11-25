""
import json
import os
import sys

seen = {}
for line in open(sys.argv[1]):
    seen.setdefault(json.loads(line)["config"], None)
for rel in seen:
    path = os.path.join(sys.argv[2], rel)
    assert os.path.isfile(path), path
    print(path)
