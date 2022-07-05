class Edge:
    def __init__(self, start,end, weight=0):
        self.start = start
        self.end = end
        self.weight = weight

    def setWeight(self, weight):
        self.weight = weight

    def setDirection(self, direction):
        self.direction = direction

    def __repr__(self) -> str:
        return (self.start.label + " => " + self.end.label)