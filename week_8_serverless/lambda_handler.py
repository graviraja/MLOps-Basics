"""
Lambda wrapper
"""

from inference_onnx import ColaONNXPredictor

inferencing_instance = ColaONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
	"""
	Lambda function handler for predicting linguistic acceptability of the given sentence
	"""
	print(event)
	return inferencing_instance.predict(event["sentence"])

