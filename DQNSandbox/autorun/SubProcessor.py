from KeyValueMapper import KeyValueMapper as mapper
from domain_layer import Training, mock_json_training, stub_training
from JSONCoder import toJSON, fromJSON
import subprocess

class Executor:

    def run(self, payload):
        url = payload.path + '/' + payload.file
        assert len(url) != 0, 'No valid url was found'
        print("exec url: %s" % url)
        dict = mapper().map(payload)
        print("dict: %s" % dict)
        subprocess.run(['python ' + url, dict])


if __name__ == "__main__":
    print("SubProcessor::main called")

    training = stub_training()

    executor = Executor()
    executor.run(training)
    print("done")