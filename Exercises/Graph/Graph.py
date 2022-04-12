from Node import Node
from Edge import Edge

class Graph:
    
    def __init__(self):
        self.nodes = {}
        self.edgesList = {}

    def addNode(self, label):
        node = Node(label)
        if label not in self.nodes.keys():
            self.nodes[label] = node

    
    def addEdge(self, From, to):
        fromNode = self.nodes[From]
        if fromNode == None:
            raise Exception("IllegalArgumentException")

        toNode = self.nodes[to]
        if toNode == None:
            raise Exception("IllegalArgumentException")

        edge = Edge(fromNode, toNode)
        key = (From, to)
        self.edgesList[key] = edge

        fromNode.neighbours[to] = edge

    
    def print(self):
        for label in self.nodes.keys():
            node = self.nodes[label]
            print("Node", label, "is connected to", node.neighbours.keys())



    def removeNode(self, label):
        node = self.nodes[label]
        if node == None:
            return

        for n in self.nodes:
            n.neigbours.remove(node)

        self.nodes.pop(label)
        

    def removeEdge(self, From, to):
        fromNode = self.nodes[From]
        toNode = self.nodes[to]

        if fromNode == None or toNode == None:
            return

        fromNode.neighbours.pop(to)
        key = (From, to)
        self.edgesList.pop(key)