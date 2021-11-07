from .nodes import Function, ActionNode, ConditionNode, SequenceNode, FallbackNode, DecoratorNode, add_child
import uuid

#############################################################################################
#############################################################################################
#############################################################################################

#############################################################################################
#############################################################################################
#############################################################################################

#############################################################################################
#############################################################################################
#############################################################################################

def controlNode(treeObj):
    node = None

    if treeObj.SUBTYPE == "SEQUENCE":
        node = SequenceNode()
    elif treeObj.SUBTYPE == "FALLBACK":
        node = FallbackNode()

    for c in treeObj.children:
        add_child(node, create_bht(c))

    return node


def decoratorNode(treeObj):
    func = None

    if treeObj.SUBTYPE == "INVERSE":
        func = Function('returnable = NodeStates.SUCCESS if status == NodeStates.SUCCESS else NodeStates.FAILED')
    elif treeObj.SUBTYPE == "FORCE_SUCCESS":
        func = Function('returnable = NodeStates.SUCCESS')
    elif treeObj.SUBTYPE == "FORCE_FAIL":
        func = Function('returnable = NodeStates.FAILED')
    elif treeObj.SUBTYPE == "LOOP":
        func = Function(f'returnable = NodeStates.RUNNING if counter < {treeObj.COUNTER} else status')

    node = DecoratorNode(func, treeObj.SUBTYPE)

    for c in treeObj.children:
        add_child(node, create_bht(c))

    return node


def actionNode(treeObj):
    node = ActionNode(Function(treeObj.CODE), treeObj.SUBTYPE)
    return node


def conditionNode(treeObj):
    node = ConditionNode(Function(treeObj.CODE, True), str(uuid.uuid4()))
    return node


#############################################################################################

nodeTypes = {
    "CONTROL": controlNode,
    "DECORATOR": decoratorNode,
    "ACTION": actionNode,
    "CONDITION": conditionNode
}

#############################################################################################

def create_bht(treeObj):
    func = nodeTypes[treeObj.TYPE]
    return func(treeObj)

def string_bht_train(tree):
    rootNode = FallbackNode()

    sequences = tree.split('|')
    for seq in sequences:
        seqNode = SequenceNode()

        conditions, actions = seq.split(';')
        
        conditions = conditions.split(',')
        for con in conditions:
            add_child(seqNode, ConditionNode(Function(f'game.passCondition("{con}")', True), str(uuid.uuid4())))

        actions = actions.split(',')
        for act in actions:
            loop, index = list(act)
            node = ActionNode(Function(f'game.play({index})'), f'{index}')

            if loop != '0':
                auxNode = DecoratorNode(Function(f'returnable = NodeStates.RUNNING if counter < {loop} else status'), 'LOOP')
                add_child(auxNode, node)
                node = auxNode

            add_child(seqNode, node)

        add_child(rootNode, seqNode)

    add_child(rootNode, ActionNode(Function('game.play(0)'), '0'))

    return rootNode

def string_bht(tree):
    rootNode = FallbackNode()

    sequences = tree.split('|')
    for seq in sequences:
        seqNode = SequenceNode()

        conditions, actions = seq.split(';')
        
        conditions = conditions.split(',')
        for con in conditions:
            add_child(seqNode, ConditionNode(Function(f'({con})', True), str(uuid.uuid4())))

        actions = actions.split(',')
        for act in actions:
            index = int(act)
            actionNum = index % 4
            loopNum = index // 4
            node = ActionNode(Function(f'observation, reward, done, info = env.step({actionNum})\nx, y, vel_x, vel_y, vel_ang, ang, l_left, l_right = observation\nfitness += reward'), act)

            if loopNum > 0:
                auxNode = DecoratorNode(Function(f'returnable = NodeStates.RUNNING if counter < {loopNum} else status'), 'LOOP')
                add_child(auxNode, node)
                node = auxNode

            add_child(seqNode, node)

        add_child(rootNode, seqNode)

    add_child(rootNode, ActionNode(Function('observation, reward, done, info = env.step(0)\nx, y, vel_x, vel_y, vel_ang, ang, l_left, l_right = observation\nfitness += reward'), '0'))

    return rootNode