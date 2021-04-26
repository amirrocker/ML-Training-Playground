from runner_class import SessionProcessor
from domain_layer import mock_json


if __name__ == "__main__":
    print("Test called")
    SessionProcessor().runSessions(mock_json())