# possible values: IntegerField, ListField, ObjectField, TimestampMillisecondsField
from pykson import JsonObject, StringField, ObjectListField
from JSONCoder import fromJSON


class Log(JsonObject):
    log = StringField()


class HyperParameter(JsonObject):
    epochs = StringField()
    learningRate = StringField()
    gamma = StringField()
    epsilon_start = StringField()
    epsilon_final = StringField()
    epsilon_decay_rate = StringField()


class ModelParameter(JsonObject):
    model = StringField()
    optimizer = StringField()
    loss = StringField()

class Training(JsonObject):
    id = StringField()
    version = StringField()
    env = StringField()
    file = StringField()
    path = StringField()
    modelParams = ModelParameter()
    log = Log()
    hyperParameters = HyperParameter()


class TrainingSession(JsonObject):
    id = StringField()
    version = StringField()
    createdAt = StringField()
    trainings = ObjectListField(Training)


# stub layer

# stub creation
def stub_training():
    json = mock_json_training()
    training = fromJSON(json, Training)
    return training


# TODO - asap replace the absolute path with a relative path

## mock
def mock_json():
    mock_json_training_session = '''
        {
            "id":"1234-ID1",
            "createdAt":"2020-04-18",
            "version":"999.9999.9999",
            "trainings":[
                {
                    "id":"1234-ID1",
                    "version":"999.9999.9999",
                    "env":"env1",
                    "file":"QPongV4.py",
                    "path":"C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/autorun",
                    "modelParams": {
                        "model":"model1",
                        "optimizer":"optimizer1",
                        "loss":"loss1"
                    },
                    "log":{
                        "log":"log1"
                    },
                    "hyperParameters":{
                        "epochs":"epochs1",
                        "learningRate":"0.00001",
                        "gamma" : "0.98",  
                        "epsilon_start" : "1.0",
                        "epsilon_final" : "0.1",
                        "epsilon_decay_rate" : "10**5"
                    }
                },
                {
                    "id":"1234-ID2",
                    "version":"999.9999.9999",
                    "env":"env2",
                    "file":"QPongV4.py",
                    "path":"C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/autorun",
                    "modelParams": {
                        "model":"model2",
                        "optimizer":"optimizer2",
                        "loss":"loss2"
                    },
                    "log":{
                        "log":"log2"
                    },
                    "hyperParameters":{
                        "epochs":"epochs2",
                        "learningRate":"0.00001",
                        "gamma" : "0.98",  
                        "epsilon_start" : "1.0",
                        "epsilon_final" : "0.1",
                        "epsilon_decay_rate" : "10**5"
                    }
                }
            ]
        }
    '''

    print(mock_json_training_session)
    return mock_json_training_session


def mock_json_training():
    return '''{
                "id":"1234-ID3",
                "version":"999.9999.9999",
                "env":"env2",
                "file":"QPongV4.py",
                "path":"C:/Users/info/Documents/Development/PycharmProjects/SelfDeepQLearning/DQNSandbox/autorun",
                "modelParams": {
                    "model":"model2",
                    "optimizer":"optimizer2",
                    "loss":"loss2"
                },
                "log":{
                    "log":"log2"
                },
                "hyperParameters":{
                    "epochs":"epochs2",
                    "learningRate":"0.00001",
                    "gamma" : "0.98",  
                    "epsilon_start" : "1.0",
                    "epsilon_final" : "0.1",
                    "epsilon_decay_rate" : "10**5"
                }
            }
    '''
