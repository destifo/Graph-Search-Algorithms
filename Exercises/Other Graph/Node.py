class Node:
    
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def __eq__(self, __o: object):
        return __o.label == self.label