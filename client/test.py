import json

from valor import Label

f = Label.key.in_(["k1", "k2", "k3"])
print(json.dumps(f.to_dict(), indent=2))
