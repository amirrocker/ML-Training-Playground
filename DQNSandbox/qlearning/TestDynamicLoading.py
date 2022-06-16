# test dynamically loading classes from modules

class Welcome:
    def welcome(self, str):
        print("Hi ho, %s says hello" % str)


class DynamicImport:
    def __init__(self):
        print("init")

    def load(self, module_name, class_name):
        module = __import__(module_name)
        clazz = getattr(module, class_name)
        # clazz.welcome(self, "Fred")
        return clazz

    def log(self, msg):
        print("self: log msg: %s" % msg)


if __name__ == "__main__":
    print("main")
    Welcome().welcome("Tom")
    dynamicImport = DynamicImport()
    obj = dynamicImport.load('TestDynamicLoading', 'Welcome')
    dynamicImport.log("sjdsldjfs")
    obj.welcome(obj, "Artie")
    another = dynamicImport.load('runner_class', 'SessionProcessor')
    # another.runSessions(another, "sessions", another)
