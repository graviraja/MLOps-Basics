"""
Lambda wrapper
"""

import json
from inference_onnx import ColaONNXPredictor

inferencing_instance = ColaONNXPredictor("./models/model.onnx")

def lambda_handler(event, context):
	"""
	Lambda function handler for predicting linguistic acceptability of the given sentence
	"""
	print(event)
	
	if "resource" in event.keys():
		http_method = event["httpMethod"]
		if http_method == "GET":
			sentence = event["queryStringParameters"]["sentence"]
			return inferencing_instance.predict(sentence)
		elif http_method == "POST":
			body = json.loads(event["body"])
			return inferencing_instance.predict(body["sentence"])
	else:
		return inferencing_instance.predict(event["sentence"])

if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	lambda_handler(test, None)
