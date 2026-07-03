import json
from json_vectorization.main import process_json

def lambda_handler(event, context):
    """
    AWS Lambda handler for processing JSON input.

    Args:
        event: The event data passed to the Lambda function. Expected keys:
               - input_json: Path to the input JSON file.
               - output_json: Path to the output JSON file.
        context: Lambda context object (not used here).

    Returns:
        dict: Result of the operation.
    """
    try:
        input_json = event.get("input_json")
        output_json = event.get("output_json")

        if not input_json or not output_json:
            return {
                "statusCode": 400,
                "body": json.dumps("Error: 'input_json' and 'output_json' parameters are required.")
            }

        process_json(input_json, output_json)

        return {
            "statusCode": 200,
            "body": json.dumps(f"Processing completed. Consolidated output saved to {output_json}.")
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
