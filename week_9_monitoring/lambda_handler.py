"""
Lambda wrapper
"""

import json
import logging
from inference_onnx import ColaONNXPredictor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

logger.info(f"Loading the model")
inferencing_instance = ColaONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
	"""
	Lambda function handler for predicting linguistic acceptability of the given sentence
	"""
	
	if "resource" in event.keys():
		body = event["body"]
		body = json.loads(body)
		logger.info(f"Got the input: {body['sentence']}")

		response = inferencing_instance.predict(body["sentence"])
		logger.info(json.dumps(response))
		return {
			"statusCode": 200,
			"headers": {},
			"body": json.dumps(response)
		}
	else:
		logger.info(f"Got the input: {event['sentence']}")
		response = inferencing_instance.predict(event["sentence"])
		logger.info(json.dumps(response))
		return response

if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	lambda_handler(test, None)
