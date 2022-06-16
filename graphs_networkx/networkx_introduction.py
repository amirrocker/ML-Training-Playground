# we may need to switch conda envs for this file
import networkx as nx


def main():
    # networkx offers four types of graphs:
    # all graph classes allow any hashable object as node*
    G = nx.Graph()  # single undirected graph. Ignores multiple edges and allows self-loop edges

    diG = nx.DiGraph()  # directed graph with directed edges

    multiG = nx.MultiGraph()  # directed graph with directed edges

    multiDiG = nx.MultiDiGraph()  # directed graph with directed edges

    # *hashable are Tuples, Strings, integers, ...
    print(G)
    print(diG)
    print(multiG)
    print(multiDiG)

    # add a simple node
    G.add_node("SomeString as Node")
    G.add_node(1)
    G.add_nodes_from([11, 12, 13, 14, 15, 16])
    G.add_nodes_from([(3, {"color": "red"})])
    G.add_nodes_from([(4, {"color": "green"})])
    G.add_nodes_from([(6, {"color": "blue"})])

    print(G)


def simpleExampleGraph():
    simple = nx.Graph()

    simple.add_nodes_from(["a", "b", "c"])

    simple.add_edge("a", "b")
    simple.add_edge("a", "c")

    # print neighbor of a
    print(simple["c"])
    print(simple["a"])
    print(len(simple["a"]))  # degree of vertex a
    print(len(simple))  # number of vertices in Graph
    print("b" in simple["c"])
    for v in simple:
        print(v)


def BFS(G, v):
    found = {v}
    waiting = [v]

    while waiting:
        w = waiting.pop(0)
        print("inspecting: {}".format(w))
        for x in G[w]:
            if x not in found:
                found.add(x)
                waiting.append(x)
                print("add: {}".format(x))
    return found


def DFS(G, v):
    found = {v}
    waiting = [v]

    while waiting:
        w = waiting.pop()
        print("inspecting: {}".format(w))
        for x in G[w]:
            if x not in found:
                found.add(x)
                waiting.append(x)
                print("add to found: {}".format(x))
    return found


def newGraph():
    G = nx.Graph()
    # letters = [choice(ascii_lowercase) for _ in range(13)]
    G.add_nodes_from(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"])

    G.add_edge("a", "b")
    G.add_edge("a", "c")
    G.add_edge("a", "d")
    G.add_edge("a", "e")

    G.add_edge("b", "f")
    G.add_edge("b", "g")

    G.add_edge("c", "h")
    G.add_edge("c", "i")

    G.add_edge("e", "j")
    G.add_edge("e", "k")
    G.add_edge("e", "l")
    G.add_edge("k", "m")
    G.add_edge("l", "m")
    return G


if (__name__ == "__main__"):
    # main()
    # simpleExampleGraph()

    G1 = newGraph()
    v = next(iter(G1))
    bfsResult = BFS(G1, v)
    print(bfsResult)

    # nx.draw(G1, with_labels=True)

    G2 = newGraph()
    v1 = next(iter(G2))
    dfsResult = DFS(G2, v)
    print(dfsResult)
    # nx.draw(G2, with_labels=True)
