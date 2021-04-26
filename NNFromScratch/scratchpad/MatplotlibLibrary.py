'''
Since plotting the data is one of the core
principles, lets take a look.
using print is nice, but it does not compare to a scatterplot or lineplot of the data.
let's start:
'''

import matplotlib.pyplot as plt

class Plot:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values


class Scatter(Plot):
    # a very simple scatter plot
    def __init__(self, x_values, y_values):
        super().__init__(x_values, y_values)
        plt.scatter(self.x_values, self.y_values)
        plt.show()


class ConfigurableScatter(Plot):
    # a very simple scatter plot
    def __init__(self, x_values, y_values):
        super().__init__(x_values, y_values)

    def withLabels(self, c, plotTitle="Title", x_label="X", y_label="y", s=40):
        print(plotTitle + " " + x_label + " " + y_label)
        if c is not None:
            print("c is not None: %s" % c)
            plt.scatter(self.x_values, self.y_values, c=self.y_values)
        else:
            plt.scatter(self.x_values, self.y_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plotTitle)
        plt.show()

class ConfigurableScatterWithColor(ConfigurableScatter):
    # a very simple scatter plot
    def __init__(self, x_values, y_values):
        super().__init__(x_values, y_values)

    def withLabelsAndColors(self, plotTitle, x_label, y_label):
        print(plotTitle + " " + x_label + " " + y_label)

        colors = {'A':'green', 'B':'yellow'}
        color_list = []
        for i in x_values:
            print("i: %s" % i)
            if i <5:
                color_list.append('green')
            else:
                color_list.append('yellow')

        plt.scatter(self.x_values, self.y_values, c=color_list)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plotTitle)
        plt.show()


'''
# Simple test section
'''

x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y_values = [1, 2, 2, 4, 3, 5, 6, 3, 2]

# Plot(x_values, y_values).scatterWithLabels("My Plot title", "xLabel value", "yLabel value")

# Scatter(x_values, y_values)

# ConfigurableScatter(x_values,y_values).withLabels("one", "two", "three")

# ConfigurableScatterWithColor(x_values, y_values).withLabelsAndColors("df", "sdf", "sdlfjs")

ConfigurableScatter(x_values, y_values).withLabels(c=y_values, s=40)

'''
This is a very simple inheritance based Scatter plot structure. But this is not ideal and has a number 
of drawbacks and flaws. Simply look at the naming to see that any further addon to the lib would lead
to a host of "XXXPlotWithYYYConfig" classes. Bad. 
Let's see what we can do with Delegation or Composition instead. A simple OO based Tip would be 
to use a Strategy or a decorator. A functional approach could be to use Monads to transform configs 
into different types of plots. (just spinning around I guess :))
'''

colors = {'A': 'green', 'B': 'yellow'}
country = ['A', 'B', 'A', 'A', 'B']
color_list = [colors[i] for i in country]
print("color_list: %s" % color_list)

for i in country:
    print(i)
