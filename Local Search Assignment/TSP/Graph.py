from collections import deque
import heapq
from Node import Node
from Edge import Edge
from math import radians, cos, sin, asin, sqrt


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
        edge2 = Edge(fromNode, toNode, weight)
        toNode.neighbours[From] = edge2
        self.edgesList[(to, From)] = edge2

    
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
        stack.append((start, 0))
        while stack:
            nod, weight = stack.pop()
            for neighbour in self.nodes[nod].neighbours.keys():
                currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                if neighbour == end:
                    path = ""
                    # path += (end + "->")
                    path += end
                    curr = nod
                    while curr != start:
                        path = curr + "->" + path
                        # path += (curr + "->")
                        curr = parentOfChild[curr]
                    # path += curr
                    path = curr + "->" + path
                    return (currEdgeWeight + weight, path)
                if neighbour not in visited:
                    parentOfChild[neighbour] = nod
                    visited.add(neighbour)
                    stack.append((neighbour, currEdgeWeight + weight))


    def bfs(self, start: str, end: str):
        node = self.nodes[start]
        if not node:
            raise Exception('Node with the given label was not found')

        visited = set()
        visited.add(start)
        queue = deque()
        parentOfChild = {} # { child:str -> parent: str} parent: the first node that spanned the child node
        queue.append((start, 0))
        while queue:
            length = len(queue)
            for i in range(length):
                nod, weight = queue.popleft()
                for neighbour in self.nodes[nod].neighbours.keys():
                    currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                    if neighbour == end:
                        path = ""
                        # path += (end + "->")
                        path += end
                        curr = nod
                        while curr != start:
                            path = curr + "->" + path
                            # path += (curr + "->")
                            curr = parentOfChild[curr]
                        # path += curr
                        path = curr + "->" + path
                        return (currEdgeWeight + weight, path)
                    if neighbour not in visited:
                        # print(neighbour)
                        parentOfChild[neighbour] = nod
                        visited.add(neighbour)
                        queue.append((neighbour, weight + currEdgeWeight))


    # returns the shortest path distance if present else -1
    def djikstraSearch(self, start: str, end: str):
        visited = set()
        minHeap = [(0, start)]

        parentOfChild = {}
        while minHeap:
            weight, nod = heapq.heappop(minHeap)
            if nod == end:
                path = ""
                # path += (end + "->")
                path += end
                curr = parentOfChild[nod]
                while curr != start:
                    path = curr + "->" + path
                    # path += (curr + "->")
                    curr = parentOfChild[curr]
                # path += curr
                path = curr + "->" + path
                # print(path)
                # print(weight)
                return (weight, path)
            visited.add(nod)
            
            for neighbour in self.nodes[nod].neighbours:
                currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                if neighbour not in visited:
                    parentOfChild[neighbour] = nod
                    heapq.heappush(minHeap, (currEdgeWeight + weight, neighbour))

        return (-1, "")


    def aStarSearch(self, start:str, end:str, heuristic_data:map):
        h = {}
        def populateHeuristicData():
            for key in heuristic_data.keys():
                h[key] = calcHeuristic(key, end)

        def calcHeuristic(initial:str, final:str):
            lon1 = radians(eval(heuristic_data[initial][1]))
            lon2 = radians(eval(heuristic_data[final][1]))
            lat1 = radians(eval(heuristic_data[initial][0]))
            lat2 = radians(eval(heuristic_data[final][0]))
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        
            c = 2 * asin(sqrt(a))
            r = 6371

            return(c * r)

        visited = set()
        populateHeuristicData()
        minHeap = [(0 + h[start], start)]
        parentOfChild = {}
        while minHeap:
            weight, nod = heapq.heappop(minHeap)
            weight = weight - h[nod]
            if nod == end:
                path = ""
                # path += (end + "->")
                path += end
                curr = parentOfChild[nod]
                while curr != start:
                    path = curr + "->" + path
                    # path += (curr + "->")
                    curr = parentOfChild[curr]
                # path += curr
                path = curr + "->" + path
                # print(path)
                # print(weight)
                return (int(weight), path)
            visited.add(nod)
            
            for neighbour in self.nodes[nod].neighbours:
                currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                if neighbour not in visited:
                    parentOfChild[neighbour] = nod
                    heapq.heappush(minHeap, (currEdgeWeight + weight + h[neighbour], neighbour))

        return -1


    def evalHeuristic(self, end:str, heuristic_data:map):
        h = {}
        def populateHeuristicData():
            for key in heuristic_data.keys():
                h[key] = calcHeuristic(key, end)

        def calcHeuristic(initial:str, final:str):
            lon1 = radians(eval(heuristic_data[initial][1]))
            lon2 = radians(eval(heuristic_data[final][1]))
            lat1 = radians(eval(heuristic_data[initial][0]))
            lat2 = radians(eval(heuristic_data[final][0]))
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        
            c = 2 * asin(sqrt(a))
            r = 6371

            return(c * r)

        populateHeuristicData()
        return h


    def aStarSearchOuterHeuristic(self, start:str, end:str, h:map):
        visited = set()
        minHeap = [(0 + h[start], start)]
        parentOfChild = {}
        while minHeap:
            weight, nod = heapq.heappop(minHeap)
            weight = weight - h[nod]
            if nod == end:
                path = ""
                # path += (end + "->")
                path += end
                curr = parentOfChild[nod]
                while curr != start:
                    path = curr + "->" + path
                    # path += (curr + "->")
                    curr = parentOfChild[curr]
                # path += curr
                path = curr + "->" + path
                # print(path)
                # print(weight)
                return (int(weight), path)
            visited.add(nod)
            
            for neighbour in self.nodes[nod].neighbours:
                currEdgeWeight = self.edgesList[(nod, neighbour)].weight
                if neighbour not in visited:
                    parentOfChild[neighbour] = nod
                    heapq.heappush(minHeap, (currEdgeWeight + weight + h[neighbour], neighbour))

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
            for j in range(i+1, N):
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