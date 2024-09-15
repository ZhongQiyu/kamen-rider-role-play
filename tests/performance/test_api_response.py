# test_api_response.py

import requests
import time
import json

class APIResponseTester:
    def __init__(self, base_url):
        self.base_url = base_url

    def send_request(self, endpoint, method='GET', data=None, headers=None):
        url = f"{self.base_url}/{endpoint}"
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return response

    def check_status_code(self, response, expected_status=200):
        return response.status_code == expected_status

    def measure_response_time(self, response):
        return response.elapsed.total_seconds()

    def validate_json_response(self, response, expected_json):
        try:
            response_json = response.json()
            return response_json == expected_json
        except json.JSONDecodeError:
            return False

    def run_tests(self, test_cases):
        for i, test_case in enumerate(test_cases):
            print(f"Running test case {i+1}: {test_case['description']}")
            
            response = self.send_request(
                endpoint=test_case['endpoint'],
                method=test_case.get('method', 'GET'),
                data=test_case.get('data'),
                headers=test_case.get('headers')
            )
            
            # Check status code
            status_ok = self.check_status_code(response, test_case['expected_status'])
            print(f"Status Code Test: {'Passed' if status_ok else 'Failed'}")
            
            # Measure response time
            response_time = self.measure_response_time(response)
            print(f"Response Time: {response_time:.4f} seconds")
            
            # Validate JSON response if expected
            if 'expected_json' in test_case:
                json_ok = self.validate_json_response(response, test_case['expected_json'])
                print(f"JSON Response Test: {'Passed' if json_ok else 'Failed'}")
            
            print("")  # Print a newline for separation between test cases

    @staticmethod
    def from_args():
        import argparse
        parser = argparse.ArgumentParser(description="Test API response")
        parser.add_argument("--base_url", type=str, required=True, help="Base URL of the API to test")
        args = parser.parse_args()
        return APIResponseTester(args.base_url)

if __name__ == "__main__":
    # Example test cases
    test_cases = [
        {
            "description": "Test GET endpoint /api/v1/resource",
            "endpoint": "api/v1/resource",
            "expected_status": 200,
            "expected_json": {"key": "value"}
        },
        {
            "description": "Test POST endpoint /api/v1/resource",
            "endpoint": "api/v1/resource",
            "method": "POST",
            "data": {"input": "test"},
            "expected_status": 201,
            "expected_json": {"result": "success"}
        }
    ]
    
    tester = APIResponseTester.from_args()
    tester.run_tests(test_cases)
