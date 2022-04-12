from Node import Node
from Edge import Edge


class Graph:
    
    def __init__(self):
        self.nodes = []
        self.nodesEdges = []
        self.edges = []
        self.edgeTuples = []

    def addNode(self, label):
        for node in self.nodes:
            if node.label == label:
                return
        node = Node(label)
        self.nodes.append(node)
        self.nodesEdges.append([])

    
    def addEdge(self, start, end):
        if start == end:
            return

        startNode = Node(start)
        endNode = Node(end)

        startIndex = -1
        endIndex = -1
        for i in range(len(self.nodes)):
            if self.nodes[i].label == startNode.label:
                startIndex = i
            if self.nodes[i].label == endNode.label:
                endIndex = i

        if startIndex == -1 or endIndex == -1:
            return

        edge = Edge(startNode, endNode)
        neighbourEdges = self.nodesEdges[startIndex]
        for neigbourEdge in neighbourEdges:
            if neigbourEdge == edge:
                return 
        
        self.edges.append(edge)
        self.edges.append((startNode, endNode))
        self.nodesEdges[startIndex].append(endNode)


    def __str__(self):
        for 

    