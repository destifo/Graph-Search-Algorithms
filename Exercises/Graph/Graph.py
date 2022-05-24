import heapq
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

    
    def addEdge(self, From, to, weight=1):
        fromNode = self.nodes[From]
        if fromNode == None:
            raise Exception("IllegalArgumentException")

        toNode = self.nodes[to]
        if toNode == None:
            raise Exception("IllegalArgumentException")

        edge = Edge(fromNode, toNode, weight)
        key = (From, to)
        self.edgesList[key] = edge

        fromNode.neighbours[to] = edge
        toNode.neighbours[From] = Edge(fromNode, toNode, weight)

    
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
        

    def removeEdge(self, From: str, to: str):
        fromNode = self.nodes[From]
        toNode = self.nodes[to]

        if fromNode == None or toNode == None:
            return

        fromNode.neighbours.pop(to)
        toNode.neighbours.pop(From)
        key = (From, to)
        self.edgesList.pop(key)


    def dfs(self, start: str, end: str):
        node = self.nodes[start]
        if node == None:
            raise Exception('Node with the given label wasn not found')

        visited = set()
        visited.add(start)
        stack = [] 
        parentOfChild = {} # { child:str -> parent: str} parent: the first node that spanned the child node
        stack.append(start)
        while stack:
            nod = stack.pop()
            for neighbour in self.nodes[nod].neighbours.keys():
                if neighbour == end:
                    path = ""
                    path += (end + ">-")
                    curr = nod
                    while curr != start:
                        path += (curr + ">-")
                        curr = parentOfChild[curr]
                    path += curr
                    print(path[::-1])
                    return
                if neighbour not in visited:
                    print(neighbour)
                    parentOfChild[neighbour] = nod
                    visited.add(neighbour)
                    stack.append(neighbour)


    def bfs(self, start: str, end: str):
        node = self.nodes[start]
        if not node:
            raise Exception('Node with the given label was not found')

        visited = set()
        visited.add(start)
        queue = []
        parentOfChild = {} # { child:str -> parent: str} parent: the first node that spanned the child node
        queue.append(start)
        while queue:
            length = len(queue)
            for i in range(length):
                nod = queue.pop()
                for neighbour in self.nodes[nod].neighbours.keys():
                    if neighbour == end:
                        path = ""
                        path += (end + ">-")
                        curr = nod
                        while curr != start:
                            path += (curr + ">-")
                            curr = parentOfChild[curr]
                        path += curr
                        print(path[::-1])
                        return
                    if neighbour not in visited:
                        print(neighbour)
                        parentOfChild[neighbour] = nod
                        visited.add(neighbour)
                        queue.append(neighbour)


    # returns the shortest path distance if present else -1
    def djikstraSearch(self, start: str, end: str):
        visited = set()
        minHeap = [(0, start)]

        parentOfChild = {}
        while minHeap:
            nod, weight = heapq.heappop(minHeap)
            visited.add(nod)
            
            for neighbour in nod.neigbours:
                currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                if neighbour == end:
                    return weight + currEdgeWeight
                if neighbour not in visited:
                    parentOfChild[neighbour.label] = nod
                    heapq.heappush(minHeap, (neighbour, currEdgeWeight + weight))

        return -1

        
    
    def toAdjacentMatrix(self):
        N = len(self.nodes)
        numToNameMap = dict()
        i = 0
        for key in self.nodes.keys():
            numToNameMap[i] = key
            i +=1
        matrix = [list(range(1 + N * i, 1 + N * (i + 1)))
                            for i in range(N)]
        
        for i in range(N):
            for j in range(i, N):
                label1 = numToNameMap[i]
                label2 = numToNameMap[j]
                node1 = self.nodes[label1]
                node2 = self.nodes[label2]
                if label2 in node1.neighbours.keys():
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

                if label1 in node2.neighbours.keys():
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

        
        return matrix