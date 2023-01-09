'''
Itch V5.0

A message protocol used to model trades and orders using around 20 message types related to system events,
stock characteristics, the placement and modification of orders.
'''
formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',
    ('alpha', 1): 's',
}