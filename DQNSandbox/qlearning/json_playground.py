'''
a = {
    "name":"Tommy",
    "age":"aNumber",
    "todo":"train"
}

b = json.dumps(a)

print(b)

var = {
    "Subjects": {
        "Math": 86,
        "Physics": 90
    }
}

print(var)

with open("sample.txt", "r") as read_it:
    data = json.load(read_it)
    print(data)
'''


class FileClass(object):

    def __init__(self, fname, mode):
        self.f_object = open(fname, mode)

    def __enter__(self):
        return self.f_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f_object.close()


class FileManagerContext():
    def __init__(self, fname, mode):
        self.filename = fname
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


'''
with FileManagerContext("sample.txt", "a") as f_write:
    f_write.write("Some more text")


print(f_write.closed)


with open("sample.txt", "w") as p:
    json.dumps(var)
'''

'''

with FileClass("sample.txt", "w") as f_open:
    f_open.write("File hase been written")
with FileClass("sample.txt", "r") as f_read:
    read = f_read.read(1024)
    print("read: %s" % read)

'''

# using pykson gson style library for python
# https://github.com/sinarezaei/pykson

# mock models setup

from pykson import JsonObject, IntegerField, StringField, ObjectListField


class Course(JsonObject):
    name = StringField()
    teacher = StringField()


class Score(JsonObject):
    score = IntegerField()
    course = Course()


class Student(JsonObject):
    first_name = StringField()
    last_name = StringField()
    age = IntegerField()
    scores = ObjectListField(Score)


# our mock models are setup

# now deserialize some json to the models

from pykson import Pykson

json_text = '{"first_name":"John", "last_name":"Smith", "age": 25, "scores": [ {"course": {"name": "Algebra", "teacher" :"Mr. Schmidt"}, "score": 100}, {"course": {"name": "Statistics", "teacher": "Mrs. Lee"}, "score": 90} ]}'
student = Pykson().from_json(json_text, Student)

print("student.age: %s and first_name: %s" % (student.age, student.first_name))
print("student.score: %s and leen(scores): %s" % (student.scores, len(student.scores)))

# second example - this time with domain models


from domain_layer import TrainingSession, Training
from runner import runSessions

# mock_json = '{"id":"1234","createdAt":"2020-04-18","trainings":[{"id":"1","env":"env1","file":"file1","modelParams": {"model":"model1","optimizer":"optimizer1","loss":"loss1"},"log":{"log":"log1"},"hyperParameters":{"epochs":"epochs1","learningRate":"learningRate1"}}]}'

mock_json = ('{\n'
             '			"id":"1234-ID",\n'
             '			"env":"env 1234-ID",\n'
             '			"file":"file 1234-ID",\n'
             '			"modelParams": {\n'
             '				"model":"model 1234-ID",\n'
             '				"optimizer":"optimizer 1234-ID",\n'
             '				"loss":"loss 1234-ID"\n'
             '			},\n'
             '			"log":{\n'
             '				"log":"log1234-ID"\n'
             '			},\n'
             '			"hyperParameters":{\n'
             '				"epochs":"epochs 1234-ID",\n'
             '				"learningRate":"learningRate 1234-ID"\n'
             '			}\n'
             '		}')

print(mock_json)

training = Pykson().from_json(mock_json, Training)
print(training.env)
print(training.file)
print(training.modelParams.loss)

''' 
training_session = Pykson().from_json(mock_json, TrainingSession)
print(training_session.id)
print(training_session.trainings[0].env)
print(len(training_session.trainings))
'''


def decode(message):
    print("callback")
    # payload = message #request.get_json()
    trainingSession = fromJSON(payload=message, clazz=TrainingSession)
    print(trainingSession)


def fromJSON(payload, clazz):
    payload.strip(" \n\r\t")
    training_session = Pykson().from_json(payload, clazz)
    return training_session


runSessions(fromJSON(mock_json, TrainingSession))
