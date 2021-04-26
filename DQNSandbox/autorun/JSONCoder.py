from pykson import Pykson

'''
using the pypi pykson library to de- and encode json.
https://pypi.org/project/pykson/
'''

def fromJSON(payload, clazz):
    payload.strip(" \n\r\t")
    assert len(payload) != 0, 'No valid JSON payload received'
    training_session = Pykson().from_json(payload, clazz)
    return training_session


def toJSON(payload):
    json = Pykson().to_json(payload)
    print("encoded json: %s" % json)
    return json


'''
if __name__ == "__main__":

    json = mock_json_training()
    obj = fromJSON(json, Training)
    str = toJSON(obj)
    print("received str: %s" % str)
'''