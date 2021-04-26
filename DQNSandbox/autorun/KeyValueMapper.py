
'''
A simple KeyValueMapper that in it first version maps
known properties on a known dictionary in a known order.
There is nothing dynamic here .... yet....
And no error checking ! so make sure what is passed in must match the code structure!
'''
class KeyValueMapper:

    def __init__(self):
        self.dict = {}

    def map(self, value):
        assert value, 'value argument not set'
        if value.id:
            self.dict['id'] = value.id
        if value.version:
            self.dict['version'] = value.version
        if value.env:
            self.dict['env'] = value.env
        if value.file:
            self.dict['file'] = value.file
        if value.path:
            self.dict['path'] = value.path

        if value.modelParams:
            if value.modelParams.model:
                self.dict['model'] = value.modelParams.model
            if value.modelParams.optimizer:
                self.dict['optimizer'] = value.modelParams.optimizer
            if value.modelParams.loss:
                self.dict['loss'] = value.modelParams.loss

        if value.hyperParameters:
            if value.hyperParameters.epochs:
                self.dict['epochs'] = value.hyperParameters.epochs
            if value.hyperParameters.learningRate:
                self.dict['learningRate'] = value.hyperParameters.learningRate
            if value.hyperParameters.gamma:
                self.dict['gamma'] = value.hyperParameters.gamma
            if value.hyperParameters.epsilon_start:
                self.dict['epsilon_start'] = value.hyperParameters.epsilon_start
            if value.hyperParameters.epsilon_final:
                self.dict['epsilon_final'] = value.hyperParameters.epsilon_final
            if value.hyperParameters.epsilon_decay_rate:
                self.dict['epsilon_decay_rate'] = value.hyperParameters.epsilon_decay_rate

        return self.dict

'''
if __name__ == "__main__":

    json = mock_json_training()
    training = fromJSON(json, Training)
    print(training.file)
    print(training.path)

    mapper = KeyValueMapper()
    result = mapper.map(training)
    print(result)

'''