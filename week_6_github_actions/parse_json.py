import json

with open('creds.txt') as f:
	data = f.read()

data = json.loads(data, strict=False)

with open('test.json', 'w') as f:
	json.dump(data, f)
