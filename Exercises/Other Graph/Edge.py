from Node import Node

class Edge:

    def __init__(self, start: Node, end: Node):
        self.start = start
        self.end = end

    def __str__(self):
        return self.start.label + " --> " + self.end.label
    
    def __eq__(self, __o: object):
        return __o.start == self.start and self.end == __o.end