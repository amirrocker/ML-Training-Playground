import pandas as pd

# def readMessageTypes():
dataFrame = pd.read_excel('message_types.xlsx').sort_values('id').drop('id', axis=1)
print(dataFrame)
# clean up
dataFrame.columns = [column.lower().strip() for column in dataFrame.columns]
print(dataFrame.columns)
dataFrame.value = dataFrame.value.str.strip()
dataFrame.name = dataFrame.name.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('/',
                                                                                                                '_')
dataFrame.notes = dataFrame.notes.str.strip()
dataFrame['message_type'] = dataFrame.loc[dataFrame.name == 'message_type', 'value']
print(dataFrame['message_type'])

messages = dataFrame.loc[:, ['message_type', 'notes']].dropna().rename(columns={'notes': 'name'})
print(messages)
messages.name = messages.name.str.lower().str.replace('message', '')
messages.name = messages.name.str.replace('.', '').str.strip().str.replace(' ', '_')
messages.to_csv('message_labels.csv', index=False)
print(messages)

dataFrame.message_type = dataFrame.message_type.ffil()
print(dataFrame)
dataFrame = dataFrame[dataFrame.name != 'message_type']
dataFrame.value = dataFrame.value.str.lower().str.replace(' ', '_').str.replace('(', '').str.strip().str.replace(')',
                                                                                                                 '')


def checkFieldCount(dataFrame):
    '''Helper function that validates the file format'''
    message_size = pd.read_excel('message_types.xlsx', sheet_name='size', index_col=0)
    message_size['check'] = dataFrame.groupby('message_type').size()
    assert message_size['size'].equals(message_size.check), 'field count does not match the template'


def checkFieldSpecification():
    messages = dataFrame.groupby('message_type')
    for t, message in messages:
        print(message.offset.add(message.length).shift().fillna(0).astype(int).equals(message.offset))


dataFrame[['message_type', 'name', 'value', 'length', 'offset', 'notes']].to_csv('message_types.csv', index=False)
