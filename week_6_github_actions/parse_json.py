import json

with open('creds.json') as f:
	data = json.load(f)

print(data)

with open('test.json', 'w') as f:
	json.dump(data, f)
