import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import numpy as np
import random
import math
import time


class SwitchBox():

    def __init__(self, position, block_type='SB'):
        self.position = position
        self.block_type = block_type

    def __hash__(self):
        return hash(self.position)
    
    def __eq__(self, other):
        return self.position == other

    def distance(self, other):
        return math.sqrt((self.position[0] - other.position[0]) ** 2 + 
                         (self.position[1] - other.position[1]) ** 2)

    def x_distance(self, other):
        return abs(self.position[0] - other.position[0])

    def y_distance(self, other):
        return abs(self.position[1] - other.position[1])

    def __str__(self):
        return '({0},{1})'.format(self.position[0], self.position[1])


def plot_graph(graph: nx.Graph, ax=None, **kwargs):
    
    positions = {node: node.position for node in graph.nodes}
    labels = {node: '{0},{1}'.format(node.position[0], node.position[1]) for node in graph.nodes}
    nx.draw(graph, ax=ax, pos=positions, node_size=NODE_SIZE, labels=labels, font_size=FONT_SIZE, **kwargs)

def generate_constrained_pointcloud():
    noc_graph = nx.Graph()
    # generate random nodes
    for i in range(0, COLUMNS):
        for j in range(0, ROWS):
            if random.random() < CHANCE:
                potential_sb = SwitchBox((i, j))
                distances = [n.distance(potential_sb) for n in noc_graph.nodes]
                if (len(distances) == 0):  # first node should be added always
                    noc_graph.add_node(SwitchBox((i, j)))
                elif (min(distances) > MINIMUM_DISTANCE):  # for later nodes check distances
                    x_distances = [n.x_distance(potential_sb) for n in noc_graph.nodes]
                    y_distances = [n.y_distance(potential_sb) for n in noc_graph.nodes]
                    if (min(x_distances) > MINIMUM_ORTHOGONAL_DISTANCE) and (min(y_distances) > MINIMUM_ORTHOGONAL_DISTANCE):
                        noc_graph.add_node(SwitchBox((i, j)))

    return noc_graph
        
def add_node_intersections(noc_graph):
    x_coordinates = {n.position[0] for n in noc_graph.nodes}
    y_coordinates = {n.position[1] for n in noc_graph.nodes}
    # add orthogonal nodes in intersections
    initial_graph = noc_graph.copy()
    for n in initial_graph.nodes:
        for x in x_coordinates:
            if DISSABLE_RANDOM or (random.random() < ORTHOGONAL_PROBABILITY):
                noc_graph.add_node(SwitchBox((x, n.position[1])))
        for y in y_coordinates:
            if DISSABLE_RANDOM or (random.random() < ORTHOGONAL_PROBABILITY):
                noc_graph.add_node(SwitchBox((n.position[0], y)))

def add_corner_nodes(noc_graph):
    max_x = max(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
    min_x = min(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
    max_y = max(noc_graph.nodes, key=lambda node: node.position[1]).position[1]
    min_y = min(noc_graph.nodes, key=lambda node: node.position[1]).position[1]

    noc_graph.add_node(SwitchBox((max_x, max_y)))
    noc_graph.add_node(SwitchBox((max_x, min_y)))
    noc_graph.add_node(SwitchBox((min_x, max_y)))
    noc_graph.add_node(SwitchBox((min_x, min_y)))

def add_edges(noc_graph):
    horizontal = [sorted([n for n in noc_graph.nodes if n.position[1] == column], key=lambda node : node.position[0]) for column in range(0, COLUMNS)]
    vertical = [sorted([n for n in noc_graph.nodes if n.position[0] == row], key=lambda node : node.position[1]) for row in range(0, ROWS)]
    for column in horizontal:
        if len(column) > 1:
            for i in range(len(column) - 1):
                weight = abs(column[i].position[0] - column[i+1].position[0])
                noc_graph.add_edge(column[i], column[i+1], weight=weight, edge_type='SYNC')
                logger.info('Added edge {0}-{1} with weight {2}'.format(str(column[i]), str(column[i+1]), str(weight)))
            weight = abs(column[-1].position[0] - column[-2].position[0])
            noc_graph.add_edge(column[-1], column[-2], weight=weight, edge_type='SYNC')  #  last one
            logger.info('Added edge {0}-{1} with weight {2}'.format(str(column[-1]), str(column[-2]), str(weight)))
 
    for row in vertical:
        if len(row) > 1:
            for i in range(len(row) - 1):
                weight = abs(row[i].position[1] - row[i+1].position[1])
                noc_graph.add_edge(row[i], row[i+1], weight=weight, edge_type='SYNC')
                logger.info('Added edge {0}-{1} with weight {2}'.format(str(row[i]), str(row[i+1]), str(weight)))
            weight = abs(row[-1].position[1] - row[-2].position[1])
            noc_graph.add_edge(row[-1], row[-2], weight=weight, edge_type='SYNC')  # last one
            logger.info('Added edge {0}-{1} with weight {2}'.format(str(row[-1]), str(row[-2]), str(weight)))

def delete_crossing_edges(noc_graph):
    initial_graph = noc_graph.copy()
    for first_edge in initial_graph.edges:
        for second_edge in initial_graph.edges:

            #print('Checking edge: ({0},{1})-({2},{3})'.format(second_edge[0].position[0], second_edge[0].position[1], second_edge[1].position[0], second_edge[1].position[1]))
            if not first_edge == second_edge:
                first_is_horizontal = first_edge[0].position[1] == first_edge[1].position[1]
                second_is_horizontal = second_edge[0].position[1] == second_edge[1].position[1]

                # only consider if they are perpendicular
                if first_is_horizontal and (not second_is_horizontal):
                    # check if they intersect
                    # they intersect if the ends that are equal are within the other ends
                    if first_is_horizontal:
                        y_coord = first_edge[0].position[1]
                        x_coord = second_edge[0].position[0]
                        max_y = max([second_edge[0].position[1], second_edge[1].position[1]])
                        min_y = min([second_edge[0].position[1], second_edge[1].position[1]])
                        max_x = max([first_edge[0].position[0], first_edge[1].position[0]])
                        min_x = min([first_edge[0].position[0], first_edge[1].position[0]])
                        #print('max_y:{0} min_y:{1}'.format(max_y, min_y))
                        # check if they intersect
                        if (y_coord > min_y) and (y_coord < max_y) and (x_coord > min_x) and (x_coord < max_x):
                            # check if tere is a node in the intersection
                            # delete the edge
                            if first_edge in noc_graph.edges:
                                noc_graph.remove_edge(*first_edge)
                                #print('Deleted edge: ({0},{1})-({2},{3})'.format(first_edge[0].position[0], first_edge[0].position[1], first_edge[1].position[0], first_edge[1].position[1]))
                    elif second_is_horizontal:
                        y_coord = second_edge[0].position[1]
                        x_coord = first_edge[0].position[0]
                        max_y = max([first_edge[0].position[1], first_edge[1].position[1]])
                        min_y = min([first_edge[0].position[1], first_edge[1].position[1]])
                        max_x = max([second_edge[0].position[0], second_edge[1].position[0]])
                        min_x = min([second_edge[0].position[0], second_edge[1].position[0]])
                        #print('max_y:{0} min_y:{1}'.format(max_y, min_y))
                        # check if they intersect
                        if (y_coord > min_y) and (y_coord < max_y) and (x_coord > min_x) and (x_coord < max_x):
                            # delete the edge
                            if second_edge in noc_graph.edges:
                                noc_graph.remove_edge(*first_edge)
                                #print('Deleted edge: ({0},{1})-({2},{3})'.format(first_edge[0].position[0], first_edge[0].position[1], first_edge[1].position[0], first_edge[1].position[1]))

def delete_midway_nodes(noc_graph):
    deleted_nodes = True 
    while deleted_nodes:
        deleted_nodes = False
        initial_graph = noc_graph.copy()
        for node in initial_graph.nodes:
            adjacents = list(noc_graph[node])
            if len(adjacents) < 2:  # node is in the end of an edge
                noc_graph.remove_node(node)
                logging.info('Deleted leaf node at' + str(node.position))
                deleted_nodes = True
            elif len(adjacents) < 3:  # means node might be in the middle or the end of an edge
                # now check if it is in the middle, this means that the adjacent nodes must be aligned
                if (adjacents[0].position[0] == adjacents[1].position[0]) or (adjacents[0].position[1] == adjacents[1].position[1]):
                    noc_graph.remove_node(node)
                    weight = max([abs(adjacents[0].position[0] - adjacents[1].position[0]), abs(adjacents[0].position[1] - adjacents[1].position[1])])
                    noc_graph.add_edge(adjacents[0], adjacents[1], weight=weight, edge_type='SYNC')
                    logging.info('Deleted midway node at' + str(node.position))
                    deleted_nodes = True


if __name__ == '__main__':
    COLUMNS = 40
    ROWS = 40
    CHANCE = 0.1
    MINIMUM_DISTANCE = 5
    MINIMUM_ORTHOGONAL_DISTANCE = 5
    NUMBER_OF_RUNS = 1
    DISSABLE_RANDOM = False
    PLOT_MARGIN = 2
    FIGURE_SIZE = 15
    NODE_SIZE = 1
    FONT_SIZE = 10
    ORTHOGONAL_PROBABILITY = 0.5
    EDGE_WIDTH = 1
    SB_COLOR = 'black'
    COLORS = {'SB': 'black', 'REG': 'blue'}
    #CLOCK_EDGE_WIDTH = 1.5
    WIRE_COLOR = 'orange'
    GRLS_COLOR = 'yellow'
    CLOCK_COLOR = 'red'
    RESOURCE_COLOR = 'xkcd:grey'
    REGISTER_COLOR = 'blue'
    #COLORS = ['green', 'orange', 'purple', 'yellow', 'pink']
    MAX_PATH_LENGHT = 4
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logging.info('Generating random pointcloud')

    for run in range(NUMBER_OF_RUNS):
        noc_graph = generate_constrained_pointcloud()
        add_node_intersections(noc_graph)
        add_corner_nodes(noc_graph)

        plot_graph(noc_graph, plt.subplot(141))
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        plt.gca().set_aspect('equal')
        
        add_edges(noc_graph)

        plot_graph(noc_graph, plt.subplot(142))
        plt.gca().set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        plt.gca().set_aspect('equal')

        delete_crossing_edges(noc_graph)
        # delete nodes with connections less than 2 (alone in the ends)
        delete_midway_nodes(noc_graph) 
        plot_graph(noc_graph, plt.subplot(143))
        plt.gca().set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        plt.gca().set_aspect('equal')
        plt.gcf().set_size_inches(13, 11)
        #plt.savefig(fname='plot' + str(run) + '.png', dpi=600)
        
        clock_tree = nx.minimum_spanning_tree(noc_graph)
        plot_graph(clock_tree, plt.subplot(144))
        plt.gca().set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        plt.gca().set_aspect('equal')
        plt.gcf().set_size_inches(13, 11)
        
        weights = nx.get_edge_attributes(noc_graph, 'weight')
        original_noc_graph = noc_graph.copy()
        for edge in original_noc_graph.edges:
            weight = weights[edge]
            if weight > MAX_PATH_LENGHT:
                noc_graph.remove_edge(*edge)
                clock_edge = edge in clock_tree.edges
                edge_type = 'SYNC' if clock_edge else 'GRLS'
                if clock_edge:
                    clock_tree.remove_edge(*edge)
                num_regs = math.ceil(weight / MAX_PATH_LENGHT) - 1
                horizontal_edge = edge[0].position[1] == edge[1].position[1]
                last_len = weight - num_regs * MAX_PATH_LENGHT
                if horizontal_edge: 
                    nodes = sorted(list(edge), key=lambda sb: sb.position[0])
                    sb = None
                    previous_sb = nodes[0]
                    for i in range(num_regs):
                        sb = SwitchBox((nodes[0].position[0] + (i + 1) * MAX_PATH_LENGHT, nodes[0].position[1]), 'REG')
                        noc_graph.add_node(sb)
                        noc_graph.add_edge(sb, previous_sb, weight=MAX_PATH_LENGHT, edge_type=edge_type)
                        clock_tree.add_edge(sb, previous_sb, weight=MAX_PATH_LENGHT, edge_type=edge_type)
                        previous_sb = sb
                    noc_graph.add_edge(nodes[1], sb, weight=last_len, edge_type=edge_type)
                    if clock_edge:
                        clock_tree.add_edge(nodes[1], sb, weight=last_len, edge_type=edge_type)
                else: 
                    nodes = sorted(list(edge), key=lambda sb: sb.position[1])
                    sb = None
                    previous_sb = nodes[0]
                    for i in range(num_regs):
                        sb = SwitchBox((nodes[0].position[0], nodes[0].position[1] + (i + 1) * MAX_PATH_LENGHT),'REG')
                        noc_graph.add_node(sb)
                        noc_graph.add_edge(sb, previous_sb, weight=MAX_PATH_LENGHT, edge_type=edge_type)
                        clock_tree.add_edge(sb, previous_sb, weight=MAX_PATH_LENGHT, edge_type=edge_type)
                        previous_sb = sb
                    noc_graph.add_edge(nodes[1], sb, weight=last_len, edge_type=edge_type)
                    if clock_edge:
                        clock_tree.add_edge(nodes[1], sb, weight=last_len, edge_type=edge_type)
        # segment long edges 
#        weights = nx.get_edge_attributes(noc_graph, 'weight')
#        original_noc_graph = noc_graph.copy()
#        for edge in original_noc_graph.edges:
#            weight = weights[edge]
#            if weight >= MAX_PATH_LENGHT:
#                horizontal_edge = edge[0].position[1] == edge[1].position[1]
#                num_regs = math.floor(weight / MAX_PATH_LENGHT) 
#                wire_len = weight / num_regs
#                if horizontal_edge:
#                    nodes = sorted(list(edge), key=lambda sb: sb.position[0])
#                    noc_graph.remove_edge(*edge)
#                    #clock_tree.remove_edge(*edge)
#                    sb = None
#                    previous_sb = nodes[0] 
#                    for i in range(num_regs):
#                        sb = SwitchBox((nodes[0].position[0] + wire_len * (i + 1), nodes[0].position[1]), 'REG')
#                        noc_graph.add_node(sb)
#                        noc_graph.add_edge(sb, previous_sb)
#                        #clock_tree.add_node(sb)
#                        #clock_tree.add_edge(sb, previous_sb)
#                        previous_sb = sb
#                        logger.info('Segmented edge: {0}-{1}'.format(str(nodes[0]), str(nodes[1])))
#                    noc_graph.add_edge(nodes[1], sb)
#                    #clock_tree.add_edge(nodes[1], sb)
#                else:
#                    nodes = sorted(list(edge), key=lambda sb: sb.position[1])
#                    noc_graph.remove_edge(*edge)
#                    #clock_tree.remove_edge(*edge)
#                    sb = None
#                    previous_sb = nodes[0] 
#                    for i in range(num_regs):
#                        sb = SwitchBox((nodes[0].position[0], nodes[0].position[1] + wire_len * (i + 1)), 'REG')
#                        noc_graph.add_node(sb)
#                        noc_graph.add_edge(sb, previous_sb)
#                        #clock_tree.add_node(sb)
#                        #clock_tree.add_edge(sb, previous_sb)
#                        previous_sb = sb
#                        logger.info('Segmented edge: {0}-{1}'.format(str(nodes[0]), str(nodes[1])))
#                    noc_graph.add_edge(nodes[1], sb)
#                    #clock_tree.add_edge(nodes[1], sb)


        floorplan_fig = plt.figure()
        ax = plt.subplot(111)
        for node in noc_graph.nodes:  
            if node.block_type == 'SB':
                x, y = node.position
                patch = pch.Rectangle((x - EDGE_WIDTH/2, y - EDGE_WIDTH/2), EDGE_WIDTH, EDGE_WIDTH, facecolor=COLORS[node.block_type], edgecolor=COLORS[node.block_type], zorder=10)
                ax.add_patch(patch)
        
        # draw the floorplan
        list_of_basis = nx.cycle_basis(noc_graph)
        list_of_basis = sorted(list_of_basis, key=lambda base : len(base))
        i = 0
        for base_cycle in list_of_basis[::-1]:
            coords = list(map(lambda node: [node.position[0], node.position[1]], base_cycle))
            #print(coords)
            color = (random.random(), random.random(), random.random())
            #patch = pch.Polygon(coords, zorder=0, color=COLORS[i])
            patch = pch.Polygon(coords, zorder=0, color=RESOURCE_COLOR)
            i += 1
            i %= len(COLORS)
            ax.add_patch(patch)

        # draw edges
        weights = nx.get_edge_attributes(noc_graph, 'weight')
        types = nx.get_edge_attributes(noc_graph, 'edge_type')
        for edge in noc_graph.edges:
            horizontal_edge = edge[0].position[1] == edge[1].position[1]
            color = WIRE_COLOR if types[edge] == 'SYNC' else GRLS_COLOR
            color = WIRE_COLOR
            if horizontal_edge:
                sorted(edge, key=lambda sw: sw.position[0])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                width = edge[1].position[0] - edge[0].position[0] + EDGE_WIDTH
                patch = pch.Rectangle(xy, width, EDGE_WIDTH, facecolor=color)
                ax.add_patch(patch)
                text_position = (edge[0].position[0] * 0.5 + edge[1].position[0] * 0.5, edge[1].position[1] - EDGE_WIDTH / 2 )
                #ax.text(*text_position, str(weights[edge]), fontsize=FONT_SIZE)

            else:  # vertical edge
                sorted(edge, key=lambda sw: sw.position[1])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                height = edge[1].position[1] - edge[0].position[1] + EDGE_WIDTH
                patch = pch.Rectangle(xy, EDGE_WIDTH, height, facecolor=color)
                ax.add_patch(patch)
                text_position = (edge[1].position[0] + EDGE_WIDTH / 2, edge[0].position[1] * 0.5 + edge[1].position[1] * 0.5)
                #ax.text(*text_position, str(weights[edge]), fontsize=FONT_SIZE)

        # draw clock edges
        for edge in clock_tree.edges:
            horizontal_edge = edge[0].position[0] == edge[1].position[0]
            if noc_graph.get_edge_data(*edge)['edge_type'] == 'SYNC':
                if horizontal_edge:
                    sorted(edge, key=lambda sw: sw.position[1])
                    xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                    height = edge[1].position[1] - edge[0].position[1] + EDGE_WIDTH
                    patch = pch.Rectangle(xy, EDGE_WIDTH / 2, height, facecolor=CLOCK_COLOR)
                    ax.add_patch(patch)
                else:  # vertical edge
                    sorted(edge, key=lambda sw: sw.position[0])
                    xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                    width = edge[1].position[0] - edge[0].position[0] + EDGE_WIDTH
                    patch = pch.Rectangle(xy, width, EDGE_WIDTH / 2, facecolor=CLOCK_COLOR)
                    ax.add_patch(patch)
        
        clock_source = list(clock_tree.nodes)[0]
        for node in clock_tree.nodes:
            clock_arrival = nx.shortest_path_length(clock_tree, clock_source, node, weight='weight')
            text_position = (node.position[0] + EDGE_WIDTH / 2 , node.position[1] + EDGE_WIDTH / 2)
            #ax.text(*text_position, clock_arrival, color='black', zorder=20, fontsize=FONT_SIZE)

        max_x = max(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
        min_x = min(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
        max_y = max(noc_graph.nodes, key=lambda node: node.position[1]).position[1]
        min_y = min(noc_graph.nodes, key=lambda node: node.position[1]).position[1]

        noc_wires = pch.Patch(color=WIRE_COLOR, label='NoC wires')
        noc_grls = pch.Patch(color=GRLS_COLOR, label='GRLS wires')
        sb = pch.Patch(color=SB_COLOR, label='Switch box')
        clock = pch.Patch(color=CLOCK_COLOR, label='Clock edge')
        resource = pch.Patch(color=RESOURCE_COLOR, label='Resource')
        register = pch.Patch(color=REGISTER_COLOR, label='Register stage')
        legend_text = [sb, noc_wires, noc_grls, clock, resource, register]
        plt.legend(loc='lower center', ncol=len(legend_text), handles=legend_text)
        ax.set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        ax.set_ylim(-PLOT_MARGIN, ROWS + PLOT_MARGIN)

        ax.set_xlim(min_x - PLOT_MARGIN, max_x + PLOT_MARGIN)
        ax.set_ylim(min_y - PLOT_MARGIN, max_y + PLOT_MARGIN)
       
        ax.set_aspect('equal')
        plt.axis('off')
        plt.gcf().set_size_inches(FIGURE_SIZE, FIGURE_SIZE)
        #plt.savefig(fname='plot' + str(run) + '.png', dpi=600, bbox_inches='tight')


        floorplan_fig = plt.figure()
        ax = plt.subplot(111)
        for node in noc_graph.nodes:
            x, y = node.position
            patch = pch.Rectangle((x - EDGE_WIDTH/2, y - EDGE_WIDTH/2), EDGE_WIDTH, EDGE_WIDTH, facecolor=COLORS[node.block_type], edgecolor=COLORS[node.block_type], zorder=10)
            ax.add_patch(patch)
        
        # draw the floorplan
        list_of_basis = nx.cycle_basis(noc_graph)
        list_of_basis = sorted(list_of_basis, key=lambda base : len(base))
        i = 0
        for base_cycle in list_of_basis[::-1]:
            coords = list(map(lambda node: [node.position[0], node.position[1]], base_cycle))
            #print(coords)
            color = (random.random(), random.random(), random.random())
            #patch = pch.Polygon(coords, zorder=0, color=COLORS[i])
            patch = pch.Polygon(coords, zorder=0, color=RESOURCE_COLOR)
            i += 1
            i %= len(COLORS)
            ax.add_patch(patch)

        # draw edges
        weights = nx.get_edge_attributes(noc_graph, 'weight')
        types = nx.get_edge_attributes(noc_graph, 'edge_type')
        for edge in noc_graph.edges:
            horizontal_edge = edge[0].position[1] == edge[1].position[1]
            color = WIRE_COLOR if types[edge] == 'SYNC' else GRLS_COLOR
            if horizontal_edge:
                sorted(edge, key=lambda sw: sw.position[0])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                width = edge[1].position[0] - edge[0].position[0] + EDGE_WIDTH
                patch = pch.Rectangle(xy, width, EDGE_WIDTH, facecolor=color)
                ax.add_patch(patch)
                text_position = (edge[0].position[0] * 0.5 + edge[1].position[0] * 0.5, edge[1].position[1] - EDGE_WIDTH / 2 )
                ax.text(*text_position, str(weights[edge]), fontsize=FONT_SIZE)

            else:  # vertical edge
                sorted(edge, key=lambda sw: sw.position[1])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                height = edge[1].position[1] - edge[0].position[1] + EDGE_WIDTH
                patch = pch.Rectangle(xy, EDGE_WIDTH, height, facecolor=color)
                ax.add_patch(patch)
                text_position = (edge[1].position[0] + EDGE_WIDTH / 2, edge[0].position[1] * 0.5 + edge[1].position[1] * 0.5)
                ax.text(*text_position, str(weights[edge]), fontsize=FONT_SIZE)

        # draw clock edges
        for edge in clock_tree.edges:
            horizontal_edge = edge[0].position[0] == edge[1].position[0]
            if horizontal_edge:
                sorted(edge, key=lambda sw: sw.position[1])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                height = edge[1].position[1] - edge[0].position[1] + EDGE_WIDTH
                patch = pch.Rectangle(xy, EDGE_WIDTH / 2, height, facecolor=CLOCK_COLOR)
                ax.add_patch(patch)
            else:  # vertical edge
                sorted(edge, key=lambda sw: sw.position[0])
                xy = (edge[0].position[0] - EDGE_WIDTH / 2, edge[0].position[1] - EDGE_WIDTH / 2)
                width = edge[1].position[0] - edge[0].position[0] + EDGE_WIDTH
                patch = pch.Rectangle(xy, width, EDGE_WIDTH / 2, facecolor=CLOCK_COLOR)
                ax.add_patch(patch)
        
        clock_source = list(clock_tree.nodes)[0]
        for node in clock_tree.nodes:
            clock_arrival = nx.shortest_path_length(clock_tree, clock_source, node, weight='weight')
            text_position = (node.position[0] + EDGE_WIDTH / 2 , node.position[1] + EDGE_WIDTH / 2)
            ax.text(*text_position, clock_arrival, color='black', zorder=20, fontsize=FONT_SIZE)

        max_x = max(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
        min_x = min(noc_graph.nodes, key=lambda node: node.position[0]).position[0]
        max_y = max(noc_graph.nodes, key=lambda node: node.position[1]).position[1]
        min_y = min(noc_graph.nodes, key=lambda node: node.position[1]).position[1]

        noc_wires = pch.Patch(color=WIRE_COLOR, label='NoC wires')
        noc_grls = pch.Patch(color=GRLS_COLOR, label='GRLS wires')
        sb = pch.Patch(color=SB_COLOR, label='Switch box')
        clock = pch.Patch(color=CLOCK_COLOR, label='Clock edge')
        resource = pch.Patch(color=RESOURCE_COLOR, label='Resource')
        register = pch.Patch(color=REGISTER_COLOR, label='Register stage')
        legend_text = [sb, noc_wires, noc_grls, clock, resource, register]
        plt.legend(loc='lower center', ncol=len(legend_text), handles=legend_text)
        ax.set_xlim(-PLOT_MARGIN, COLUMNS + PLOT_MARGIN)
        ax.set_ylim(-PLOT_MARGIN, ROWS + PLOT_MARGIN)

        ax.set_xlim(min_x - PLOT_MARGIN, max_x + PLOT_MARGIN)
        ax.set_ylim(min_y - PLOT_MARGIN, max_y + PLOT_MARGIN)
       
        ax.set_aspect('equal')
        plt.axis('off')
        plt.gcf().set_size_inches(FIGURE_SIZE, FIGURE_SIZE)
        #plt.savefig(fname='plot' + str(run) + '.png', dpi=600, bbox_inches='tight')

        plt.show()

        #print(len(nx.cycle_basis(noc_graph)))
        print('  Total length of NoC:          {0}'.format(noc_graph.size(weight='weight')))
        print('  Lengh of clock tree:          {0}'.format(clock_tree.size(weight='weight')))
        print('  Number of GRLS stages:        {0}'.format(len([e for e in noc_graph.edges if e not in clock_tree.edges])))
        print('  Number of registrer stages:   {0}'.format(len([n for n in noc_graph.nodes if n.block_type == 'REG'])))
