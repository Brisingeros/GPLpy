from enum import Enum

class NodeStates(Enum):
    SUCCESS = 1
    RUNNING = 2
    FAILED = 3
    FORCE_END = 4

##############################################################

class Node:
    def __init__(self):
        self.children = None

    def tick(self, data):
        return NodeStates.FAILED

    def printTree(self, root, markerStr="+--- ", levelMarkers=[]):
        emptyStr = " "*len(markerStr)
        connectionStr = "|" + emptyStr[:-1]
        level = len(levelMarkers)
        mapper = lambda draw: connectionStr if draw else emptyStr
        markers = "".join(map(mapper, levelMarkers[:-1]))
        markers += markerStr if level > 0 else ""
        
        treeShow = f"{markers}{root.name}"
        
        for i, child in enumerate(root.children):
            isLast = i == len(root.children) - 1
            treeShow += '\n' + self.printTree(child, markerStr, [*levelMarkers, not isLast])

        return treeShow

    def __str__(self):
        return f'{self.printTree(self)}'

##############################################################

class Function:
    def __init__(self, code, check = False):
        if check:
            self._code = f'returnable = NodeStates.SUCCESS if {code} else NodeStates.FAILED'
        else:
            self._code = code
        
        self.code = self._code

    def run(self, data, status = None):
        data['done'] = False
        if data['done']:
            return NodeStates.FORCE_END

        data['status'] = status
        data['NodeStates'] = NodeStates
        data['returnable'] = NodeStates.SUCCESS
        exec(self.code, globals(), data)

        if data['done']:
            return NodeStates.FORCE_END

        return data['returnable']

##############################################################
##############################################################
##############################################################

class FunctionNode(Node):
    def __init__(self, function, name = ''):
        super().__init__()

        if not (isinstance(function, Function)):
            raise Exception('function is not Function object')

        self.function = function
        self.name = name

    def tick(self, data):
        try:
            self.function.run(data)
            return NodeStates.SUCCESS
        except:
            return super().tick(data)

class ActionNode(FunctionNode):
    def __init__(self, function, name):
        super().__init__(function, f'ACTION_{name}')

    def tick(self, data):
        try:
            return self.function.run(data)
        except:
            return super().tick(data)

class ConditionNode(FunctionNode):
    def __init__(self, function, name):
        super().__init__(function, f'CONDITION_{name}')

    def tick(self, data):
        try:
            return self.function.run(data)
        except:
            return super().tick(data)

##############################################################

class ControlNode(Node):
    def __init__(self):
        super().__init__()
        self.children = []

    def tick(self, data):
        return super().tick(data)

class SequenceNode(ControlNode):
    def __init__(self):
        super().__init__()

        self.name = '-->'

    def tick(self, data):
        status = NodeStates.SUCCESS
        i = 0
        try:
            while (i < len(self.children) and status == NodeStates.SUCCESS):
                child = self.children[i]

                status = child.tick(data)
                while (status == NodeStates.RUNNING):
                    status = child.tick(data)

                i += 1

            return status
        except:
            return super().tick(data)

class FallbackNode(ControlNode):
    def __init__(self):
        super().__init__()

        self.name = '?'

    def tick(self, data):
        status = NodeStates.FAILED
        i = 0
        try:
            while (i < len(self.children) and status == NodeStates.FAILED):
                child = self.children[i]
                
                status = child.tick(data)
                while (status == NodeStates.RUNNING):
                    status = child.tick(data)

                i += 1

            return status
        except:
            return super().tick(data)

##############################################################

class DecoratorNode(FunctionNode):
    def __init__(self, function, name):
        super().__init__(function, f'DECORATOR_{name}')
        self.counter = 0

    def tick(self, data):
        try:
            status = NodeStates.RUNNING

            while (status == NodeStates.RUNNING):
                status = self.children.tick(data)
                
                data['counter'] = self.counter
                status = self.function.run(data, status)

                self.counter += 1

            self.counter = 0
            return status
        except:
            return super().tick(data)

##############################################################

def add_child(node, node_c):
    if not (isinstance(node_c, Node)):
        raise Exception('Child not assignable')

    if (isinstance(node, ControlNode)):
        node.children.append(node_c)
        return
    elif (isinstance(node, DecoratorNode)):
        node.children = node_c
        return

    raise Exception('Child not assignable')
