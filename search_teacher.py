#!/usr/bin/env python3
"""
search_teacher.py - An interactive tool for learning Depth-First Search (DFS)
and Breadth-First Search (BFS).

Students can:
  * choose DFS, BFS, or compare both
  * pick any start and goal node
  * watch every step: node expanded, frontier (stack/queue), visited set
  * edit the graph and try their own examples

Run:  python3 search_teacher.py
Requires Python 3.6+ (standard library only).

Key idea shown by the trace
---------------------------
Both algorithms are IDENTICAL except for how the frontier is managed:
    DFS -> frontier is a STACK  (LIFO: take the newest node)
    BFS -> frontier is a QUEUE  (FIFO: take the oldest node)
Neighbours are always explored in alphabetical order so results are
reproducible and easy to check by hand.
"""

# --------------------------------------------------------------------------
# Default example graph (undirected). Students can replace it from the menu.
#
#            A
#          / | \
#         B  C  D
#        /|  |   \
#       E F  G    H
#       |    |\    \
#       I    J K    L
# --------------------------------------------------------------------------
DEFAULT_EDGES = [
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "E"), ("B", "F"),
    ("C", "G"),
    ("D", "H"),
    ("E", "I"),
    ("G", "J"), ("G", "K"),
    ("H", "L"),
]


def build_graph(edges):
    """Turn a list of (u, v) pairs into an undirected adjacency dictionary."""
    graph = {}
    for u, v in edges:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    return {node: sorted(neigh) for node, neigh in sorted(graph.items())}


def print_graph(graph):
    print("\nCurrent graph (adjacency list):")
    for node, neigh in graph.items():
        print(f"  {node} -> {', '.join(neigh)}")
    print()


# --------------------------------------------------------------------------
# The search itself
# --------------------------------------------------------------------------
def search(graph, start, goal, method, verbose=True):
    """
    Run DFS or BFS from start to goal.

    method  : "DFS" or "BFS"
    Returns : (path or None, expansion_order)

    The frontier holds (node, parent) pairs. A node is marked visited when
    it is removed from the frontier ("expanded"); if it is already visited
    we simply skip it. The goal test happens at expansion time.
    """
    frontier = [(start, None)]
    visited = []              # order in which nodes were expanded
    parent = {}               # used to rebuild the path at the end
    step = 0

    if verbose:
        name = "stack (LIFO, top is on the RIGHT)" if method == "DFS" \
            else "queue (FIFO, front is on the LEFT)"
        print(f"\n=== {method} from {start} to {goal} ===")
        print(f"Frontier is a {name}\n")
        print(f"{'Step':<5}{'Expand':<8}{'Frontier after step':<42}Visited")
        print("-" * 78)

    while frontier:
        # DFS pops the NEWEST item (end), BFS pops the OLDEST item (front)
        node, par = frontier.pop() if method == "DFS" else frontier.pop(0)

        if node in visited:
            if verbose:
                print(f"      (skip {node}: already visited)")
            continue

        step += 1
        visited.append(node)
        parent[node] = par

        if node == goal:
            if verbose:
                print(f"{step:<5}{node:<8}{'GOAL FOUND!':<42}{' '.join(visited)}")
            return rebuild_path(parent, goal), visited

        # Add unvisited neighbours to the frontier
        new = [n for n in graph[node] if n not in visited]
        if method == "DFS":
            # Push in reverse so the alphabetically-first neighbour is on top
            for n in reversed(new):
                frontier.append((n, node))
        else:
            for n in new:
                frontier.append((n, node))

        if verbose:
            shown = "[" + ", ".join(n for n, _ in frontier) + "]"
            print(f"{step:<5}{node:<8}{shown:<42}{' '.join(visited)}")

    return None, visited


def rebuild_path(parent, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def report(method, path, visited):
    if path:
        print(f"\n{method} result: path found -> {' -> '.join(path)}")
        print(f"  Path length : {len(path) - 1} edge(s)")
    else:
        print(f"\n{method} result: NO path exists.")
    print(f"  Nodes expanded: {len(visited)}  ({' '.join(visited)})")


# --------------------------------------------------------------------------
# User interaction helpers
# --------------------------------------------------------------------------
def ask_node(graph, prompt):
    while True:
        value = input(prompt).strip().upper()
        if value in graph:
            return value
        print(f"  '{value}' is not in the graph. Choose from: {' '.join(graph)}")


def ask_method():
    while True:
        print("Choose an algorithm:")
        print("  1) Depth-First Search   (DFS)")
        print("  2) Breadth-First Search (BFS)")
        print("  3) Compare both")
        choice = input("Your choice [1/2/3]: ").strip().upper()
        if choice in ("1", "DFS"):
            return "DFS"
        if choice in ("2", "BFS"):
            return "BFS"
        if choice in ("3", "BOTH", "C"):
            return "BOTH"
        print("  Please enter 1, 2 or 3.\n")


def edit_graph(graph):
    print("\nEnter edges one per line as  X-Y  (e.g. A-B). Names are")
    print("single words, case-insensitive. Blank line when finished.")
    print("Leave it empty to keep the current graph.")
    edges = []
    while True:
        line = input("  edge> ").strip().upper()
        if not line:
            break
        parts = [p.strip() for p in line.split("-")]
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            print("  Format must be X-Y with two different nodes.")
            continue
        edges.append((parts[0], parts[1]))
    if edges:
        return build_graph(edges)
    return graph


def explain_comparison(graph, start, goal, dfs_res, bfs_res):
    (dp, dv), (bp, bv) = dfs_res, bfs_res
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"{'':<14}{'DFS':<32}{'BFS'}")
    print(f"{'Path':<14}{(' -> '.join(dp) if dp else 'none'):<32}"
          f"{(' -> '.join(bp) if bp else 'none')}")
    print(f"{'Path length':<14}{(len(dp) - 1 if dp else '-'):<32}"
          f"{(len(bp) - 1 if bp else '-')}")
    print(f"{'Expanded':<14}{len(dv):<32}{len(bv)}")
    print()
    if dp and bp:
        if len(dp) > len(bp):
            print("Notice: DFS found a LONGER path. DFS goes deep first and takes")
            print("the first route it finds. BFS explores level by level, so it")
            print("always finds a path with the fewest edges.")
        elif len(dp) == len(bp):
            print("Both found a shortest path this time. The default graph is a tree,")
            print("so every path is unique. Use [E] to edit the graph and add a")
            print("shortcut or cycle (e.g. A-H, or C-F) to see DFS and BFS differ.")
        if len(dv) < len(bv):
            print("DFS expanded fewer nodes here, but that is luck of the layout,")
            print("not a guarantee. Try a different goal.")
        elif len(dv) > len(bv):
            print("BFS expanded fewer nodes here (the goal was close to the start).")
    print("\nMemory: BFS must store a whole level in its queue (can be huge);")
    print("DFS only stores the current path plus siblings (usually much less).")


# --------------------------------------------------------------------------
# Main program
# --------------------------------------------------------------------------
def main():
    graph = build_graph(DEFAULT_EDGES)
    print("=" * 78)
    print("   DFS & BFS Teaching Tool")
    print("=" * 78)

    while True:
        print_graph(graph)
        method = ask_method()
        start = ask_node(graph, "Start node: ")
        goal = ask_node(graph, "Goal node : ")

        if method == "BOTH":
            dfs_res = search(graph, start, goal, "DFS")
            report("DFS", *dfs_res)
            bfs_res = search(graph, start, goal, "BFS")
            report("BFS", *bfs_res)
            explain_comparison(graph, start, goal, dfs_res, bfs_res)
        else:
            path, visited = search(graph, start, goal, method)
            report(method, path, visited)

        print("\nWhat next?")
        print("  [Enter] run another search   [E] edit the graph   [Q] quit")
        nxt = input("> ").strip().upper()
        if nxt == "Q":
            print("Goodbye - happy searching!")
            break
        if nxt == "E":
            graph = edit_graph(graph)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
