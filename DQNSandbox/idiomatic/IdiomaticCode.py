import time

colors = ['red', 'green', 'blue']

# looping
start = time.thread_time_ns()
for i in [0, 1, 2, 3, 4, 5]:
    print(i)

for i in range(0, 100):
    print("range: %d" % i**2)

for i in range(60000):
    print(i**3)
    for i, color in enumerate(colors):
        print("color i[%d]: %s" % (i, color))

end = time.thread_time_ns()
time_spent = (end - start) / 1000000 / 1000 # time_spent in seconds
print("time spent: %.3f(s), start: %.3f , end: %.3f" % (time_spent, start, end))
