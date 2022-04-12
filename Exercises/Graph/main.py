from Graph import Graph


graph = Graph()
graph.addNode('a')
graph.addNode('b')
graph.addNode('c')

graph.addEdge('a', 'b')
graph.addEdge('a', 'c')

graph.print()
print()
for k, v in graph.edgesList.items():
    print(k, "=>", v)
