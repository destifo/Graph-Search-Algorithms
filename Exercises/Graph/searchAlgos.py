def dfs(self, start, end):
        node = self.nodes[start]
        if node == None:
            raise Exception('Node with the given label wasn not found')

        isVisited = set()
        stack = []
        stack.append(start)
        while stack:
            for neighbour in self.nodes[stack.pop()].neighbours.keys():
                if neighbour == end:
                    return
                if neighbour not in isVisited:
                    print(neighbour)
                    isVisited.add(neighbour)
                    stack.append(neighbour)


def bfs(self, start, end):
    node = self.nodes[start]
    if not node:
        raise Exception('Node with the given label was not found')

    isVisited = set()
    queue = []
    queue.append(start)
    while queue:
        for neighbour in self.nodes[queue.pop(0)].neighbours.keys():
            if neighbour == end:
                return
            if neighbour not in isVisited:
                print(neighbour)
                isVisited.add(neighbour)
                queue.append(neighbour)