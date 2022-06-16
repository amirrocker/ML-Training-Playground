from runner_class import SessionProcessor


## mock
def mockJSON():
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

    mock_json_2 = '''
        {
            "id":"1234-ID1",
            "createdAt":"2020-04-18",
            "version":"999.9999.9999",
            "trainings":[
                {
                    "id":"1234-ID1",
                    "version":"999.9999.9999",
                    "env":"env1",
                    "file":"file1",
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
                        "learningRate":"learningRate1"
                    }
                },
                {
                    "id":"1234-ID2",
                    "version":"999.9999.9999",
                    "env":"env2",
                    "file":"file2",
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
                        "learningRate":"learningRate2"
                    }
                }
            ]
        }
    '''

    print(mock_json_2)
    return mock_json_2


if __name__ == "__main__":
    print("Test called")
    conn = "conn"
    SessionProcessor(conn).runSessions(mockJSON(), conn)
