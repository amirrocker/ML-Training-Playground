# A short primer on import, packages, modules and resources in python. As well as dynamically loading
# python files at runtime which is what we are interested in.
# see here: https://realpython.com/python-import/

# start with import
import math
print(math.pi)

# importing math imports the code of the math module.
# math is not only an imported module, it also represents a namespace.
# that is why we write math.pi not just pi.
# list the contents of a namespace with dir()
print(dir(math))
# or the global namespace
print(dir())

# most modern languages allow naming of imports
import math as myMath
print(myMath.pi)

# and also selectively import
from math import acos as myAcosFunction
print(myAcosFunction(0.7))

# TODO Note the technical side note: the module namespace is
#  implemented as a python dict. Use
#  math.__dict__["pi"] will print 3.14159......

