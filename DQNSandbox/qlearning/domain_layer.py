# possible values: IntegerField, ListField, ObjectField, TimestampMillisecondsField
from pykson import JsonObject, StringField, ObjectListField

class Log(JsonObject):
    log = StringField()


class HyperParameter(JsonObject):
    epochs = StringField()
    learningRate = StringField()


class ModelParameter(JsonObject):
    model = StringField()
    optimizer = StringField()
    loss = StringField()


class Training(JsonObject):
    id = StringField()
    env = StringField()
    file = StringField()
    modelParams = ModelParameter()
    log = Log()
    hyperParameters = HyperParameter()


class TrainingSession(JsonObject):
    id = StringField()
    trainingID = StringField()
    createdAt = StringField()
    trainings = ObjectListField(Training)