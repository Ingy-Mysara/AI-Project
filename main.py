from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from queue import PriorityQueue
from collections import deque

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def main():
    searchAlgorithmsGUI()


def searchAlgorithmsGUI():
    global root
    root = Tk()
    root.title("Search Algorithms")

    global f
    f = plt.figure(figsize=(5, 5))

    global a
    a = f.add_subplot(111)
    plt.axis('off')

    global G
    G = nx.DiGraph()

    global canvas
    canvas = FigureCanvasTkAgg(f, master=root)

    global heuristic
    heuristic = {}

    global heuristic2
    heuristic2 = {}

    arr = []
    arrG = []

    global graphastar
    graphastar = {}

    global graphgreedy
    graphgreedy = {}

    button_clear = Button(root, text="clear", padx=10, command=clear_btn)
    button_clear.grid(column=7, row=0, columnspan=1, padx=10, pady=10)

    label_start = Label(text="Start Node").grid(padx=10, pady=10, row=0, column=1)
    label_end = Label(text="End Node").grid(padx=10, pady=10, row=0, column=2)
    label_weight = Label(text="Weight/Limit").grid(padx=10, pady=10, row=0, column=3)
    label_h1 = Label(text="Start Node h()").grid(padx=10, pady=10, row=0, column=4)
    label_h2 = Label(text="End Node h()").grid(padx=10, pady=10, row=0, column=5)

    global label_nodes
    label_nodes = Label(root, text="Nodes")
    label_nodes.grid(column=0, row=8, padx=10, pady=10)

    global cost_label
    cost_label = Label(root, text="")
    cost_label.grid(column=1, row=8, padx=10, pady=10)

    # BFS
    tb_node1_bfs = Entry(root, width=5)
    tb_node1_bfs.grid(padx=10, pady=10, row=1, column=1)
    tb_node2_bfs = Entry(root, width=5)
    tb_node2_bfs.grid(padx=10, pady=10, row=1, column=2)
    button_in_bfs = Button(root, text="input", padx=10,
                           command=lambda: bfs_input(tb_node1_bfs.get(), tb_node2_bfs.get()))
    button_in_bfs.grid(column=6, row=1, columnspan=1, padx=10, pady=10)
    button_bfs = Button(root, text="Breadth First Search", padx=10,
                        command=lambda: bfs(tb_node1_bfs.get(), tb_node2_bfs.get()))
    button_bfs.grid(column=7, row=1, columnspan=1, padx=10, pady=10)

    # DFS
    tb_node1_dfs = Entry(root, width=5)
    tb_node1_dfs.grid(padx=10, pady=10, row=2, column=1)
    tb_node2_dfs = Entry(root, width=5)
    tb_node2_dfs.grid(padx=10, pady=10, row=2, column=2)
    button_in_dfs = Button(root, text="input", padx=10,
                           command=lambda: dfs_input(tb_node1_dfs.get(), tb_node2_dfs.get()))
    button_in_dfs.grid(column=6, row=2, columnspan=1, padx=10, pady=10)
    button_dfs = Button(root, text="Depth First Search", padx=10,
                        command=lambda: dfs(tb_node1_dfs.get(), tb_node2_dfs.get()))
    button_dfs.grid(column=7, row=2, columnspan=1, padx=10, pady=10)

    # UCS
    tb_node1_ucs = Entry(root, width=5)
    tb_node1_ucs.grid(padx=10, pady=10, row=3, column=1)
    tb_node2_ucs = Entry(root, width=5)
    tb_node2_ucs.grid(padx=10, pady=10, row=3, column=2)
    tb_weight_ucs = Entry(root, width=5)
    tb_weight_ucs.grid(padx=10, pady=10, row=3, column=3)
    button_in_ucs = Button(root, text="input", padx=10,
                           command=lambda: ucs_input(tb_node1_ucs.get(), tb_node2_ucs.get(), tb_weight_ucs.get()))
    button_in_ucs.grid(column=6, row=3, columnspan=1, padx=10, pady=10)
    button_ucs = Button(root, text="Uniform Cost Search", padx=10,
                        command=lambda: ucs(tb_node1_ucs.get(), tb_node2_ucs.get()))
    button_ucs.grid(column=7, row=3, columnspan=1, padx=10, pady=10)

    # DLS
    tb_node1_dls = Entry(root, width=5)
    tb_node1_dls.grid(padx=10, pady=10, row=4, column=1)
    tb_node2_dls = Entry(root, width=5)
    tb_node2_dls.grid(padx=10, pady=10, row=4, column=2)
    tb_limit_dls = Entry(root, width=5)
    tb_limit_dls.grid(padx=10, pady=10, row=4, column=3)
    button_in_dls = Button(root, text="input", padx=10,
                           command=lambda: dls_input(tb_node1_dls.get(), tb_node2_dls.get()))
    button_in_dls.grid(column=6, row=4, columnspan=1, padx=10, pady=10)
    button_dls = Button(root, text="Depth Limited Search", padx=10,
                        command=lambda: dls(tb_node1_dls.get(), tb_node2_dls.get(), tb_limit_dls.get()))
    button_dls.grid(column=7, row=4, columnspan=1, padx=10, pady=10)

    # Iterative
    tb_node1_iterative = Entry(root, width=5)
    tb_node1_iterative.grid(padx=10, pady=10, row=5, column=1)
    tb_node2_iterative = Entry(root, width=5)
    tb_node2_iterative.grid(padx=10, pady=10, row=5, column=2)
    button_in_iterative = Button(root, text="input", padx=10,
                                 command=lambda: iterative_input(tb_node1_iterative.get(), tb_node2_iterative.get()))
    button_in_iterative.grid(column=6, row=5, columnspan=1, padx=10, pady=10)
    button_iterative = Button(root, text="Iterative Deepening Search", padx=10,
                              command=lambda: iterative(tb_node1_iterative.get(), tb_node2_iterative.get()))
    button_iterative.grid(column=7, row=5, columnspan=1, padx=10, pady=10)

    # Greedy
    tb_node1_greedy = Entry(root, width=5)
    tb_node1_greedy.grid(padx=10, pady=10, row=6, column=1)
    tb_node2_greedy = Entry(root, width=5)
    tb_node2_greedy.grid(padx=10, pady=10, row=6, column=2)
    tb_h1_greedy = Entry(root, width=5)
    tb_h1_greedy.grid(padx=10, pady=10, row=6, column=4)
    tb_h2_greedy = Entry(root, width=5)
    tb_h2_greedy.grid(padx=10, pady=10, row=6, column=5)
    button_in_greedy = Button(root, text="input", padx=10,
                              command=lambda: greedy_input(tb_node1_greedy.get(), tb_node2_greedy.get(),
                                                           tb_h1_greedy.get(), tb_h2_greedy.get()))
    button_in_greedy.grid(column=6, row=6, columnspan=1, padx=10, pady=10)
    button_greedy = Button(root, text="Greedy Search", padx=10,
                           command=lambda: greedy(tb_node1_greedy.get(), tb_node2_greedy.get()))
    button_greedy.grid(column=7, row=6, columnspan=1, padx=10, pady=10)

    # A*
    tb_node1_astar = Entry(root, width=5)
    tb_node1_astar.grid(padx=10, pady=10, row=7, column=1)
    tb_node2_astar = Entry(root, width=5)
    tb_node2_astar.grid(padx=10, pady=10, row=7, column=2)
    tb_weight_astar = Entry(root, width=5)
    tb_weight_astar.grid(padx=10, pady=10, row=7, column=3)
    tb_h1_astar = Entry(root, width=5)
    tb_h1_astar.grid(padx=10, pady=10, row=7, column=4)
    tb_h2_astar = Entry(root, width=5)
    tb_h2_astar.grid(padx=10, pady=10, row=7, column=5)
    button_in_astar = Button(root, text="input", padx=10,
                             command=lambda: astar_input(tb_node1_astar.get(), tb_node2_astar.get(),
                                                         tb_weight_astar.get(), tb_h1_astar.get(), tb_h2_astar.get()))
    button_in_astar.grid(column=6, row=7, columnspan=1, padx=10, pady=10)
    button_astar = Button(root, text="A* Search", padx=10,
                          command=lambda: astar(tb_node1_astar.get(), tb_node2_astar.get()))
    button_astar.grid(column=7, row=7, columnspan=1, padx=10, pady=10)

    root.mainloop()


def clear_btn():
    G.clear()
    s = "[]"
    label_nodes.config(text=s)
    graphastar.clear()
    graphgreedy.clear()
    heuristic.clear()
    heuristic2.clear()

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def bfsHelper(graph, start, goal, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    queue = []
    queue.append(start)
    path = []
    while queue:
        s = queue.pop(0)
        path.append(s)
        if s in goal:
            return path
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return path


def bfsH2(start, goal, graph):
    queue = deque([start])
    visited = {start: None}

    while queue:
        cur_node = queue.popleft()
        if cur_node == goal:
            break

        next_nodes = G[cur_node]
        for next_node in next_nodes:
            if next_node not in visited:
                queue.append(next_node)
                visited[next_node] = cur_node
    return visited


def bfs(start, goal):
    path = bfsHelper(G, start, goal)
    # print(path)
    path2 = []
    visited = bfsH2(start, goal, G)
    # print(visited)
    cur_node = goal
    print(f'\npath from {goal} to {start}: \n {goal} ', end='')
    path2.append(goal)
    while cur_node != start:
        cur_node = visited[cur_node]
        path2.append(cur_node)
    node_colors = ['black' if not node in path else 'lightcoral' if node in path2 else 'deepskyblue' for node in
                   G.nodes()]
    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def bfs_input(node1, node2):
    G.add_edges_from([(node1, node2)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def dfspath(graph, root, goal):
    visited = []
    fringe = [[root]]
    while fringe:
        path = fringe.pop(-1)
        node = path[-1]
        if node not in visited:
            children = G.neighbors(node)
            for child in children:
                new_path = list(path)
                new_path.append(child)
                fringe.append(new_path)
                if child == goal:
                    return new_path
                visited.append(node)
    return "no path exists"


def dfs(start, goal):
    path = dfsHelper(G, start, goal)
    path2 = dfs_helper2(G, start, goal)
    node_colors = ['black' if not node in path else 'lightcoral' if node in path2 else 'deepskyblue' for node in
                   G.nodes()]

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def dfsHelper(graph, start, goal, path=None):
    if path is None:
        path = []
    path_list = [[start]]
    visited = []

    while path_list:

        path = path_list.pop()
        print(path)
        last_node = path[-1]
        visited.append(last_node)
        if last_node == goal:
            print(path)
            return visited

        else:
            for node in graph[last_node]:
                if node not in path:
                    new_path = path + [node]
                    print(new_path)
                    path_list.append(new_path)
                    # print(path_list)


def dfs_input(node1, node2):
    G.add_edges_from([(node1, node2)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def dfs_helper2(G, start, goal):
    path_list = [[start]]

    while path_list:

        path = path_list.pop()

        last_node = path[-1]
        if last_node == goal:
            return path

        else:
            for node in G[last_node]:
                if node not in path:
                    new_path = path + [node]
                    path_list.append(new_path)

    print('No path exists between %s and %s' % (start, goal))


def get_cost(G, from_node, to_node):
    cost = G.get_edge_data(from_node, to_node)
    return int(cost["weight"])


def ucsHelper(G, start, goal):
    visited = []
    queue = PriorityQueue()
    queue.put((0, start))
    total_cost = 0
    while queue:
        cost, node = queue.get()
        if node not in visited:
            visited.append(node)

            if node == goal:
                return visited
            for i in G.neighbors(node):
                if i not in visited:
                    total_cost = cost + get_cost(G, node, i)
                    queue.put((total_cost, i))
                cost_label.config(text=("Total Cost:", total_cost))
                print("total cost: ", total_cost)


def ucs(node1, node2):
    path = ucsHelper(G, node1, node2)
    print(path)
    node_colors = ['black' if not node in path else 'lightcoral' for node in G.nodes()]
    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def ucs_input(node1, node2, weight):
    G.add_weighted_edges_from([(node1, node2, weight)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx(G, pos, ax=a)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def dls(start, goal, maxD):
    path = list()
    p = DLS_helper(start, goal, path, maxD)
    node_colors = ['black' if not node in p else 'lightcoral' for node in G.nodes()]

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def DLS_helper(start, goal, path, level_of_search, current_level=0):
    lvl_int = int(level_of_search)

    path.append(start)
    if start == goal:
        return path

    if current_level == lvl_int:
        return False

    for child in G[start]:
        if DLS_helper(child, goal, path, current_level + 1, lvl_int):
            return path
        path.pop()

    return False


def dls_input(node1, node2):
    G.add_edges_from([(node1, node2)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def iterative(start, goal, maxD=9999):
    path = []
    p = iterative_helper(start, goal, path, maxD)
    node_colors = ['black' if not node in p else 'lightcoral' for node in G.nodes()]

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)

    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def iterative_helper(start, goal, path, maxD):
    maxD_int = int(maxD)

    path.append(start)
    if start == goal:
        return path

    if maxD_int <= 0:
        return False

    for child in G[start]:
        if iterative_helper(child, goal, path, maxD_int - 1):
            return path
        path.remove(child)
    return False


def iterative_input(node1, node2):
    G.add_edges_from([(node1, node2)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def greedy(start, goal):
    global graphgreedy, heuristic2
    path = []
    path.append(start)
    min_node = start
    minV = 9999
    while minV != 0:
        for node in graphgreedy[min_node]:
            print("node ", node)
            s = str(node)
            num = int(heuristic2[s])
            if num < minV:
                minV = int(heuristic2[s])
                min_node = s
        path.append(min_node)
    node_colors = ['black' if not node in path else 'lightcoral' for node in G.nodes()]

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def greedy_input(node1, node2, h1, h2):
    G.add_edges_from([(node1, node2)])
    s = str(G.nodes)
    label_nodes.config(text=s)
    heuristic2[node1] = h1
    heuristic2[node2] = h2
    global arrG
    if node1 in graphgreedy:
        arrG.append(node2)
    else:
        arrG = []
        arrG.append(node2)
    graphgreedy[node1] = arrG
    print(graphgreedy)

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)


def astar(start, goal):
    cost = {start: 0}
    closed = []
    opened = [[start, heuristic[start]]]

    # find the visited nodes
    while True:
        fn = [i[1] for i in opened]
        chosen_index = fn.index(min(fn))
        node = opened[chosen_index][0]
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == goal:
            break
        for item in graphastar[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:
                continue
            cost.update({item[0]: int(int(cost[node]) + int(item[1]))})
            fn_node = int(cost[node]) + int(heuristic[item[0]]) + int(item[1])
            temp = [item[0], int(fn_node)]
            opened.append(temp)

            # find optimal sequence
    trace_node = goal
    optimal_sequence = [goal]
    for i in range(len(closed) - 2, -1, -1):
        check_node = closed[i][0]  # current node
        if trace_node in [children[0] for children in graphastar[check_node]]:
            children_costs = [temp[1] for temp in graphastar[check_node]]
            children_nodes = [temp[0] for temp in graphastar[check_node]]

            # whether h(s) + g(s) = f(s). If so, append current node to optimal sequence
            # change the correct optimal tracing node to current node
            if int(cost[check_node]) + int(children_costs[children_nodes.index(trace_node)]) == int(cost[trace_node]):
                optimal_sequence.append(check_node)
                trace_node = check_node
    optimal_sequence.reverse()  # reverse the optimal sequence
    print("closed")
    print(closed)
    print("optimal")
    print(optimal_sequence)

    print("node")
    for node in G.nodes():
        print(node)

    node_colors = ['black' if not node in optimal_sequence else 'lightcoral' for node in G.nodes()]

    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Reds'), node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)
    print(optimal_sequence)


def astar_input(node1, node2, weight, h1, h2):
    G.add_weighted_edges_from([(node1, node2, weight)])
    s = str(G.nodes)
    label_nodes.config(text=("Nodes in Graph: ", s))
    heuristic[node1] = h1
    heuristic[node2] = h2
    global arr
    if node1 in graphastar:
        arr2 = []
        arr2.append(node2)
        arr2.append(weight)
        arr.append(arr2)
    else:
        arr2 = []
        arr = []
        arr2.append(node2)
        arr2.append(weight)
        arr.append(arr2)

    graphastar[node1] = arr
    plt.clf()
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.get_tk_widget().grid(padx=10, rowspan=8, row=0, column=0, columnspan=1)
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=600)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    print(G.nodes)
    print(heuristic)
    print(graphastar)


main()