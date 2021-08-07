import json

with open('creds.txt') as f:
	data = f.read()

print(data)
# data = json.loads(data, strict=False)
# print(data)
data = eval(data)
print(data)

with open('test.json', 'w') as f:
	json.dump(data, f)
