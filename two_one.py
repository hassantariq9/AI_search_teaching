#!/usr/bin/env python3
"""
Searching demo: the Two-One problem and the Tower of Hanoi
===========================================================

Both puzzles are written as *search problems* with the same four components
from the lecture:

    1. Problem statement   (the rules / constraints)
    2. Goal state          (what "solved" looks like)
    3. Solution space      (every state reachable from the start state)
    4. Operators           (actions that move us from one state to another)

A single generic search engine (BFS and DFS) solves both problems, and
matplotlib draws what is happening:

    * the solution, move by move
    * the whole solution space (state-space graph) with dead ends and the
      solution path highlighted
    * a comparison of search strategies (BFS vs DFS vs random hit-and-trial)
    * an animated GIF of the solution

Requirements:  Python 3.8+,  matplotlib  (pillow is needed only for GIFs)
    pip install matplotlib pillow

Examples:
    python search_demo.py                  # no size given -> asks YOU for the number of coins
                                           #   per side (Two-One interactive session)
    python search_demo.py -i               # same, explicitly
    python search_demo.py --problem twoone --coins 4   # 4 Rs.1 coins vs 4 Rs.2 coins
    python search_demo.py --problem twoone --start 11_222     # your own start state
    python search_demo.py --problem twoone --starts 1_2 11_22 111_222
    python search_demo.py --problem twoone --sweep 5   # sizes 1..5 + growth chart
    python search_demo.py --problem hanoi --disks 4
    python search_demo.py --problem both               # both puzzles with defaults
    python search_demo.py --show --gif                 # open windows + save GIFs

Interactive Two-One mode:  first you are asked how many coins per side you want.
Then at every prompt you can
    <number>  solve that many coins per side (3 -> 111_222)
    +         add ONE MORE coin to each side and solve again (or just press Enter)
    s         type your own start state (characters 1, 2 and _ ; e.g. 11_222 or 1__22)
    t         results table so far
    g         growth chart (needs 2+ standard sizes)
    q         quit
The goal state is always the mirror image of the start state (the two kinds of
coin swap sides).
"""

import argparse
import math
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------- #
#  Colours (kept in one place so the figures look consistent)
# --------------------------------------------------------------------------- #
NAVY, TEAL, AMBER, BLUE = "#0B1E3D", "#127C8A", "#E8A33D", "#2C5EA8"
GREEN, RED, GREY, LIGHT = "#3E9B5A", "#C24A4A", "#8FA3C4", "#F5F7FB"

MAX_COINS = 9          # per side; larger puzzles have millions of states


# =========================================================================== #
#  1.  A tiny generic search framework
# =========================================================================== #
class Problem:
    """Base class: subclasses define the start state, goal test and operators."""

    initial: Hashable

    def is_goal(self, state) -> bool:
        raise NotImplementedError

    def actions(self, state) -> List[Tuple[str, Hashable]]:
        """Return a list of (operator_label, next_state) pairs."""
        raise NotImplementedError


@dataclass
class SearchResult:
    algorithm: str
    solved: bool
    states: List[Hashable]          # start ... goal
    moves: List[str]                # operator used between consecutive states
    expanded: int                   # how many states were expanded
    generated: int                  # how many distinct states were discovered
    order: List[Hashable] = field(default_factory=list)   # expansion order


def _build_result(name, parent, goal, order) -> SearchResult:
    states, moves, s = [], [], goal
    while s is not None:
        states.append(s)
        link = parent[s]
        if link is None:
            break
        moves.append(link[1])
        s = link[0]
    states.reverse()
    moves.reverse()
    return SearchResult(name, True, states, moves, len(order), len(parent), order)


def search(problem: Problem, algorithm: str = "bfs") -> SearchResult:
    """Graph search. 'bfs' uses a queue (shortest solution), 'dfs' a stack."""
    start = problem.initial
    parent = {start: None}
    frontier = deque([start])
    order: List[Hashable] = []
    while frontier:
        state = frontier.popleft() if algorithm == "bfs" else frontier.pop()
        order.append(state)
        if problem.is_goal(state):
            return _build_result(algorithm.upper(), parent, state, order)
        for label, nxt in problem.actions(state):
            if nxt not in parent:
                parent[nxt] = (state, label)
                frontier.append(nxt)
    return SearchResult(algorithm.upper(), False, [], [], len(order), len(parent), order)


def explore_everything(problem: Problem):
    """Breadth-first sweep of the WHOLE solution space.

    Returns (depth_of_each_state, list_of_edges) where an edge is
    (state, next_state, label).
    """
    depth = {problem.initial: 0}
    edges = []
    queue = deque([problem.initial])
    while queue:
        s = queue.popleft()
        for label, t in problem.actions(s):
            edges.append((s, t, label))
            if t not in depth:
                depth[t] = depth[s] + 1
                queue.append(t)
    return depth, edges


def hit_and_trial(problem: Problem, trials: int = 5000, max_steps: int = 500, seed: int = 1):
    """'Generate and test' without any memory: random legal moves until we
    either reach the goal, get stuck, or give up.
    Returns (solved, stuck, gave_up, average_moves_when_solved)."""
    rng = random.Random(seed)
    solved = stuck = gave_up = 0
    solved_steps = 0
    for _ in range(trials):
        s = problem.initial
        for _step in range(max_steps):
            if problem.is_goal(s):
                solved += 1
                solved_steps += _step
                break
            options = problem.actions(s)
            if not options:
                stuck += 1
                break
            s = rng.choice(options)[1]
        else:
            gave_up += 1
    avg = solved_steps / solved if solved else float("nan")
    return solved, stuck, gave_up, avg


# =========================================================================== #
#  2.  The Two-One problem
# =========================================================================== #
class TwoOne(Problem):
    """Rs.1 coins on the left, Rs.2 coins on the right, and at least one empty slot.

    State   : a string such as '11_22'  ('_' is an empty slot)
    Start   : any string over {1, 2, _} with at least one '_'
              (TwoOne(3) is shorthand for the standard start '111_222')
    Goal    : the mirror image of the start, e.g. '22_11' (the coins swap sides)
    Rules   : Rs.1 coins move only to the right, Rs.2 coins only to the left,
              one coin moves per turn.
    Operators:
        Slide - move a coin into an adjacent empty slot
        Hop   - jump over exactly one coin into an empty slot
    """

    def __init__(self, start=2):
        if isinstance(start, int):
            start = "1" * start + "_" + "2" * start
        self.initial = parse_start(start)
        self.goal = self.initial[::-1]
        self.ones = self.initial.count("1")
        self.twos = self.initial.count("2")
        self.n = self.ones if self.ones == self.twos else max(self.ones, self.twos)

    def describe(self) -> str:
        return f"{self.ones} Rs.1 + {self.twos} Rs.2 coins, start {self.initial}"

    def is_goal(self, s) -> bool:
        return s == self.goal

    def actions(self, s):
        out, size = [], len(s)
        for i, coin in enumerate(s):
            step = 1 if coin == "1" else -1 if coin == "2" else 0
            if step == 0:
                continue
            near, far = i + step, i + 2 * step
            if 0 <= near < size and s[near] == "_":
                out.append((f"Slide Rs.{coin} {'right' if step > 0 else 'left'}", self._move(s, i, near)))
            elif 0 <= far < size and s[near] != "_" and s[far] == "_":
                out.append((f"Hop Rs.{coin} {'right' if step > 0 else 'left'}", self._move(s, i, far)))
        return out

    @staticmethod
    def _move(s, i, j):
        lst = list(s)
        lst[j], lst[i] = lst[i], "_"
        return "".join(lst)


def parse_start(text: str) -> str:
    """Validate a user-typed Two-One start state (characters 1, 2 and _ only)."""
    s = text.strip().replace(" ", "").replace(".", "_").replace("-", "_")
    if not s or any(c not in "12_" for c in s):
        raise ValueError("use only the characters 1, 2 and _  (example: 11_222)")
    if "_" not in s:
        raise ValueError("there must be at least one empty slot '_'")
    if "1" not in s and "2" not in s:
        raise ValueError("put at least one coin (1 or 2) in the start state")
    if len(s) > 17:
        raise ValueError("start state too long (max 17 slots)")
    return s


def draw_coins(ax, state, x0=0.0, y0=0.0, slot=1.0, border=GREY, lw=1.5):
    """Draw one Two-One state (a row of slots with coins) on `ax`."""
    n = len(state)
    ax.add_patch(Rectangle((x0, y0), n * slot, slot, fill=True, fc="white", ec=border, lw=lw))
    for i, c in enumerate(state):
        if i:
            ax.plot([x0 + i * slot] * 2, [y0, y0 + slot], color="#C7D4EA", lw=0.8)
        if c != "_":
            ax.add_patch(Circle((x0 + (i + 0.5) * slot, y0 + slot / 2), 0.38 * slot,
                                fc=AMBER if c == "1" else BLUE, ec="none"))
            ax.text(x0 + (i + 0.5) * slot, y0 + slot / 2, c, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=NAVY if c == "1" else "white")


def plot_two_one_solution(problem: TwoOne, res: SearchResult, path: str):
    """Draw the solution move by move (wraps into several columns if it is long)."""
    k, per = len(res.states), 12
    ncols = math.ceil(k / per)
    size = len(res.states[0])
    colw = size + 5.8
    rows = min(k, per)
    xmin, xmax, ymin, ymax = -2.6, ncols * colw - 0.5, -0.4, rows * 1.5
    unit = 0.55
    fig, ax = plt.subplots(figsize=((xmax - xmin) * unit, (ymax - ymin) * unit + 0.6))
    for i, s in enumerate(res.states):
        col, row = divmod(i, per)
        x0, y = col * colw, (rows - 1 - row) * 1.5
        label = "Start" if i == 0 else f"Move {i}"
        ax.text(x0 - 0.5, y + 0.5, label, ha="right", va="center", fontsize=10, fontweight="bold", color=NAVY)
        draw_coins(ax, s, x0, y, border=GREEN if s == problem.goal else (BLUE if i == 0 else GREY),
                   lw=2.5 if i in (0, k - 1) else 1.5)
        if i:
            ax.text(x0 + size + 0.4, y + 0.5, res.moves[i - 1], va="center", fontsize=10, color=NAVY)
            if row:
                ax.annotate("", xy=(x0 + size / 2, y + 1.0), xytext=(x0 + size / 2, y + 1.45),
                            arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Two-One ({problem.describe()}): {res.algorithm} solution in {len(res.moves)} moves",
                 fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_two_one_state_space(problem: TwoOne, res: SearchResult, path: str):
    """The complete solution space, drawn in columns by distance from the start."""
    depth, edges = explore_everything(problem)
    levels: Dict[int, List[str]] = {}
    for s, d in depth.items():
        levels.setdefault(d, []).append(s)
    parents: Dict[str, List[str]] = {}
    for a, b, _ in edges:
        if depth[b] > depth[a]:
            parents.setdefault(b, []).append(a)

    pos = {}
    for d in sorted(levels):                       # barycentre ordering reduces edge crossings
        items = levels[d]
        if d:
            items.sort(key=lambda s: sum(pos[p][1] for p in parents[s]) / len(parents[s]))
        m = len(items)
        for i, s in enumerate(items):
            pos[s] = (d, (i - (m - 1) / 2) * -1.3 if m > 1 else 0.0)

    sol_edges = set(zip(res.states, res.states[1:]))
    dead = {s for s in depth if not problem.actions(s) and not problem.is_goal(s)}
    width = max(12, 1.6 * (max(levels) + 1))
    height = max(5.5, 0.95 * max(len(v) for v in levels.values()) + 2)
    fig, ax = plt.subplots(figsize=(width, height))

    for a, b, label in edges:
        if depth[b] <= depth[a]:
            continue
        on_path = (a, b) in sol_edges
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=GREEN if on_path else (AMBER if label.startswith("Hop") else GREY),
                lw=3.2 if on_path else 1.1, zorder=1 if not on_path else 2)
    for s, (x, y) in pos.items():
        face = "#E7F3EA" if problem.is_goal(s) else "#EAF1FC" if s == problem.initial else "#FBEAEA" if s in dead else "white"
        edge = GREEN if problem.is_goal(s) else BLUE if s == problem.initial else RED if s in dead else GREY
        ax.text(x, y, " ".join(s), ha="center", va="center", fontsize=9.5, family="monospace", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=face, ec=edge, lw=1.6), zorder=3)

    ax.set_xlim(-0.6, max(levels) + 0.6)
    ax.set_ylim(min(p[1] for p in pos.values()) - 1, max(p[1] for p in pos.values()) + 1)
    ax.axis("off")
    ax.set_title(f"Two-One solution space ({problem.initial}): {len(depth)} states, {len(dead)} dead ends "
                 f"(green = solution, grey = slide, amber = hop, red box = dead end)",
                 fontsize=11, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_comparison(problem: Problem, name: str, path: str, trials: int = 5000):
    """Left: how much work each strategy does. Right: what random hit-and-trial achieves."""
    bfs_r, dfs_r = search(problem, "bfs"), search(problem, "dfs")
    solved, stuck, gave_up, avg = hit_and_trial(problem, trials=trials)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    names = ["BFS", "DFS"]
    expanded = [bfs_r.expanded, dfs_r.expanded]
    length = [len(bfs_r.moves), len(dfs_r.moves)]
    x = range(2)
    b1 = a1.bar([i - 0.2 for i in x], expanded, 0.4, color=TEAL, label="states expanded")
    b2 = a1.bar([i + 0.2 for i in x], length, 0.4, color=AMBER, label="solution length (moves)")
    for b in list(b1) + list(b2):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())), ha="center", va="bottom", fontsize=10)
    a1.set_xticks(list(x))
    a1.set_xticklabels(names)
    a1.set_title("Systematic search", color=NAVY, fontweight="bold")
    a1.legend(frameon=False)
    a1.spines[["top", "right"]].set_visible(False)

    labels = ["solved", "stuck", "gave up"]
    vals = [solved, stuck, gave_up]
    cols = [GREEN, RED, GREY]
    bars = a2.bar(labels, [100 * v / trials for v in vals], color=cols)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{100 * v / trials:.1f}%", ha="center", va="bottom", fontsize=10)
    a2.set_ylim(0, 112)
    a2.set_ylabel("% of random trials")
    a2.set_title(f"Hit-and-trial ({trials:,} random runs)", color=NAVY, fontweight="bold")
    if solved:
        a2.text(0.98, 0.93, f"when it succeeds it needs ~{avg:.0f} moves\n(shortest possible: {len(bfs_r.moves)})",
                transform=a2.transAxes, ha="right", va="top", fontsize=9.5, color=NAVY)
    a2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{name}: why we need systematic search", fontsize=13, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig, (bfs_r, dfs_r, (solved, stuck, gave_up, avg))


# =========================================================================== #
#  3.  Tower of Hanoi
# =========================================================================== #
class Hanoi(Problem):
    """n disks on 3 pegs. State = (peg0, peg1, peg2); each peg is a tuple of
    disk sizes from bottom to top (bigger number = bigger disk).

    Goal     : all disks on the last peg, in the same order.
    Rules    : move one disk at a time; never place a larger disk on a smaller one.
    Operator : move the top disk of peg i to peg j.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.initial = (tuple(range(n, 0, -1)), (), ())
        self.goal = ((), (), tuple(range(n, 0, -1)))

    def is_goal(self, s) -> bool:
        return s == self.goal

    def actions(self, s):
        out = []
        for i in range(3):
            if not s[i]:
                continue
            disk = s[i][-1]
            for j in range(3):
                if i != j and (not s[j] or s[j][-1] > disk):
                    pegs = [list(p) for p in s]
                    pegs[i].pop()
                    pegs[j].append(disk)
                    out.append((f"Move disk {disk}: {'ABC'[i]} \u2192 {'ABC'[j]}", tuple(tuple(p) for p in pegs)))
        return out


def hanoi_recursive(n, src="A", aux="B", dst="C"):
    """The classic recursive solution: 2**n - 1 moves."""
    if n == 0:
        return []
    return hanoi_recursive(n - 1, src, dst, aux) + [(n, src, dst)] + hanoi_recursive(n - 1, aux, src, dst)


def draw_hanoi(ax, state, n, title=None):
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.3, n + 1.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0.2, -0.25), 8.6, 0.25, fc=NAVY, ec="none"))
    cmap = plt.get_cmap("viridis")
    for p in range(3):
        cx = 1.5 + 3 * p
        ax.add_patch(Rectangle((cx - 0.06, 0), 0.12, n + 1.0, fc="#7B8794", ec="none"))
        ax.text(cx, -0.65, "ABC"[p], ha="center", fontsize=11, fontweight="bold", color=NAVY)
        for level, disk in enumerate(state[p]):
            w = 0.5 + 2.2 * disk / n
            ax.add_patch(FancyBboxPatch((cx - w / 2, level + 0.04), w, 0.82,
                                        boxstyle="round,pad=0,rounding_size=0.25",
                                        fc=cmap(0.15 + 0.7 * (disk - 1) / max(1, n - 1)), ec="white", lw=1.2))
            ax.text(cx, level + 0.45, str(disk), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=9, color=NAVY)


def plot_hanoi_solution(problem: Hanoi, res: SearchResult, path: str):
    k = len(res.states)
    cols = min(4, k)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows))
    axes = list(axes.flat) if k > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < k:
            draw_hanoi(ax, res.states[i], problem.n,
                       "Start" if i == 0 else f"{i}. {res.moves[i - 1]}")
        else:
            ax.axis("off")
    fig.suptitle(f"Tower of Hanoi ({problem.n} disks): {res.algorithm} solution in {len(res.moves)} moves "
                 f"(optimal = 2^{problem.n} \u2212 1 = {2 ** problem.n - 1})", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def plot_hanoi_state_space(problem: Hanoi, res: SearchResult, path: str):
    """The 3**n states form a Sierpinski triangle - draw the whole solution space."""
    depth, edges = explore_everything(problem)
    n = problem.n
    corners = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    centroid = (0.5, math.sqrt(3) / 6)

    def position(state):
        where = {}                                   # disk -> peg
        for p, peg in enumerate(state):
            for d in peg:
                where[d] = p
        sigma = [0, 1, 2]                            # peg -> corner of the current triangle
        x = y = 0.0
        for k, d in enumerate(range(n, 0, -1)):     # biggest disk decides the big triangle
            p = where[d]
            cx, cy = corners[sigma[p]]
            x += 2 ** -(k + 1) * cx
            y += 2 ** -(k + 1) * cy
            # Inside this sub-triangle, "smaller disks all on peg q" sits at the corner that
            # points toward the cluster for the THIRD peg (that is where the big disk can hop).
            sigma = [sigma[p] if q == p else sigma[3 - p - q] for q in range(3)]
        return x + 2 ** -n * centroid[0], y + 2 ** -n * centroid[1]

    pos = {s: position(s) for s in depth}
    sol_edges = set(zip(res.states, res.states[1:])) | set(zip(res.states[1:], res.states))
    fig, ax = plt.subplots(figsize=(8, 7.2))
    for a, b, _ in edges:
        on = (a, b) in sol_edges
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=GREEN if on else GREY,
                lw=3.2 if on else 0.9, zorder=2 if on else 1)
    size = 70 if n <= 3 else 22 if n == 4 else 8
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=size, c="white", ec=NAVY, zorder=3)
    for s, c, lab in [(problem.initial, BLUE, "START\n(all on A)"), (problem.goal, GREEN, "GOAL\n(all on C)")]:
        ax.scatter(*pos[s], s=size * 3, c=c, zorder=4)
        ax.annotate(lab, pos[s], xytext=(0, -26), textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=c)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Tower of Hanoi solution space: {len(depth)} states = 3^{n}\n(green = shortest solution)",
                 fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# =========================================================================== #
#  4.  Animation
# =========================================================================== #
def animate_solution(states, moves, draw_fn, title, path=None, show=False, interval=900, figsize=(7, 3.4)):
    """Animate a list of states. Saves a GIF if `path` is given."""
    fig, ax = plt.subplots(figsize=figsize)

    def frame(i):
        ax.clear()
        draw_fn(ax, states[i])
        sub = "Start" if i == 0 else f"Move {i}/{len(moves)}: {moves[i - 1]}"
        ax.set_title(f"{title}\n{sub}", fontsize=11, color=NAVY, fontweight="bold")

    anim = FuncAnimation(fig, frame, frames=len(states), interval=interval, repeat=True)
    if path:
        try:
            anim.save(path, writer="pillow", fps=max(1, round(1000 / interval)))
            print(f"   saved {path}")
        except Exception as exc:                    # pillow missing, etc.
            print(f"   (could not write GIF: {exc})")
    if show:
        plt.show()
    return anim


# =========================================================================== #
#  5.  Putting it together
# =========================================================================== #
def report(res: SearchResult, extra=""):
    print(f"   {res.algorithm}: {'solved' if res.solved else 'NO SOLUTION'} | "
          f"moves = {len(res.moves)} | states expanded = {res.expanded} | "
          f"states discovered = {res.generated} {extra}")


STATE_SPACE_DRAW_LIMIT = 80       # only draw the full state-space graph when it is small enough


def solve_two_one(start, args, out, draw=True, trials=3000):
    """Solve one Two-One start state with BFS and DFS, print the result, save figures.
    Returns a record (dict) used for the summary table and the growth chart."""
    prob = TwoOne(start)
    print(f"\n=== Two-One problem: {prob.describe()}  (goal {prob.goal}) ===")
    depth, _ = explore_everything(prob)
    dead = sum(1 for s in depth if not prob.actions(s) and not prob.is_goal(s))
    bfs_r, dfs_r = search(prob, "bfs"), search(prob, "dfs")
    res = bfs_r if args.algo == "bfs" else dfs_r
    print(f"   Solution space: {len(depth)} reachable states, {dead} dead ends")
    report(bfs_r)
    report(dfs_r)
    rec = dict(start=prob.initial, ones=prob.ones, twos=prob.twos, states=len(depth), dead=dead,
               solved=res.solved, moves=len(res.moves), bfs_exp=bfs_r.expanded, dfs_exp=dfs_r.expanded,
               dfs_moves=len(dfs_r.moves))
    if not res.solved:
        print("   NO SOLUTION exists for this start state (every path ends in a dead end).")
    elif not res.moves:
        print("   The start state is already the goal.")
    else:
        print(f"   {res.algorithm} solution ({len(res.moves)} moves):")
        for i, (m, s) in enumerate(zip(res.moves, res.states[1:]), 1):
            print(f"     {i:2d}. {m:<18} -> {s}")
    if draw:
        tag = prob.initial.replace("_", "-")
        if res.solved and res.moves:
            plot_two_one_solution(prob, res, os.path.join(out, f"two_one_{tag}_solution.png"))
        if len(depth) <= STATE_SPACE_DRAW_LIMIT:
            plot_two_one_state_space(prob, res, os.path.join(out, f"two_one_{tag}_state_space.png"))
        else:
            print(f"   (state-space drawing skipped: {len(depth)} states is too many to draw legibly)")
        _, (_, _, rnd) = plot_comparison(prob, f"Two-One {prob.initial}", os.path.join(out, f"two_one_{tag}_compare.png"), trials=trials)
        print(f"   Random hit-and-trial: solved {rnd[0]}, stuck {rnd[1]}, gave up {rnd[2]} of {trials} runs")
        if (args.gif or args.show) and res.solved and res.moves:
            animate_solution(res.states, res.moves, lambda ax, s: _coins_frame(ax, s), f"Two-One {prob.initial}",
                             os.path.join(out, f"two_one_{tag}_solution.gif") if args.gif else None, args.show)
        plt.close("all")                           # free memory in long interactive sessions
    return rec


def print_table(records):
    if not records:
        print("   (no runs yet)")
        return
    head = f"{'start state':<18}{'Rs.1':>5}{'Rs.2':>6}{'states':>8}{'dead':>6}{'BFS moves':>11}{'BFS exp.':>10}{'DFS exp.':>10}"
    print("\n" + head + "\n" + "-" * len(head))
    for r in records:
        mv = r["moves"] if r["solved"] else "none"
        print(f"{r['start']:<18}{r['ones']:>5}{r['twos']:>6}{r['states']:>8}{r['dead']:>6}{str(mv):>11}{r['bfs_exp']:>10}{r['dfs_exp']:>10}")


def plot_growth(records, path, show=False):
    """How the puzzle grows as we add coins (standard starts: n Rs.1 vs n Rs.2)."""
    std = {}
    for r in records:
        if r["ones"] == r["twos"] and r["solved"] and r["start"] == "1" * r["ones"] + "_" + "2" * r["twos"]:
            std[r["ones"]] = r
    if len(std) < 2:
        print("   (growth chart needs at least two different standard sizes, e.g. 2 and 3 coins per side)")
        return None
    ns = sorted(std)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.semilogy(ns, [std[n]["states"] for n in ns], "o-", color=TEAL, lw=2.5, label="reachable states")
    a1.semilogy(ns, [std[n]["bfs_exp"] for n in ns], "s--", color=AMBER, lw=2, label="states BFS expanded")
    for n in ns:
        a1.annotate(str(std[n]["states"]), (n, std[n]["states"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a1.set_xlabel("coins per side (n)")
    a1.set_title("The solution space explodes", color=NAVY, fontweight="bold")
    a1.set_xticks(ns)
    a1.legend(frameon=False)
    a1.spines[["top", "right"]].set_visible(False)
    a2.plot(ns, [std[n]["moves"] for n in ns], "o-", color=GREEN, lw=2.5, label="shortest solution (BFS)")
    a2.plot(ns, [n * n + 2 * n for n in ns], ":", color=NAVY, lw=2, label="formula n\u00B2 + 2n")
    for n in ns:
        a2.annotate(str(std[n]["moves"]), (n, std[n]["moves"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a2.set_xlabel("coins per side (n)")
    a2.set_title("...but the best solution grows slowly", color=NAVY, fontweight="bold")
    a2.set_xticks(ns)
    a2.legend(frameon=False)
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Two-One problem: adding coins", fontsize=13, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"   saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def ask_coins(default: int) -> int:
    """Ask the user how many coins per side (Enter = default)."""
    while True:
        try:
            raw = input(f"How many coins per side? [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= MAX_COINS:
            return int(raw)
        print(f"  Please type a whole number from 1 to {MAX_COINS}.")


def interactive_two_one(args, out):
    """Menu-driven session: ask for coins per side, then keep adding coins."""
    print("\n" + "=" * 64)
    print(" Two-One problem - INTERACTIVE MODE")
    print("=" * 64)
    n = ask_coins(max(1, args.coins))               # the user chooses the starting size
    records = [solve_two_one(n, args, out)]
    print("\nAt the prompt you can type:")
    print("  <number>  coins per side, e.g. 3  ->  start 111_222")
    print("  +         add one more coin to each side (also: just press Enter)")
    print("  s         type your own start state (characters 1, 2, _)")
    print("  t         table of results so far")
    print("  g         growth chart (needs 2+ standard sizes)")
    print("  q         quit")
    while True:
        try:
            raw = input(f"\nYour choice [+ = {min(n + 1, MAX_COINS)} coins per side]: ").strip().lower() or "+"
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            if raw in ("q", "quit", "exit"):
                break
            elif raw == "t":
                print_table(records)
            elif raw == "g":
                plot_growth(records, os.path.join(out, "two_one_growth.png"), args.show)
            elif raw == "s":
                start = parse_start(input("  Start state (e.g. 11_222, 1__22, 112_2): "))
                records.append(solve_two_one(start, args, out))
            elif raw == "+" or raw.isdigit():
                n = n + 1 if raw == "+" else int(raw)
                if n < 1:
                    raise ValueError("need at least 1 coin per side")
                if n > MAX_COINS:
                    print(f"  {n} coins per side is very large (millions of states); capping at {MAX_COINS}.")
                    n = MAX_COINS
                records.append(solve_two_one(n, args, out, draw=True))
            else:
                print("  Please type a number, +, s, t, g or q.")
        except ValueError as exc:
            print(f"  Invalid input: {exc}")
    print("\nSession summary:")
    print_table(records)
    plot_growth(records, os.path.join(out, "two_one_growth.png"), False)
    return records


def run_two_one(args, out):
    if args.interactive:
        return interactive_two_one(args, out)
    records = []
    if args.sweep:
        for n in range(1, args.sweep + 1):
            records.append(solve_two_one(n, args, out, draw=(n <= 3)))
    elif args.starts:
        for s in args.starts:
            records.append(solve_two_one(s, args, out))
    else:
        start = args.start if args.start else args.coins
        records.append(solve_two_one(start, args, out))
    if len(records) > 1:
        print_table(records)
        plot_growth(records, os.path.join(out, "two_one_growth.png"), args.show)
    return records


def _coins_frame(ax, state):
    draw_coins(ax, state, 0, 0, slot=1.0)
    ax.set_xlim(-0.3, len(state) + 0.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def run_hanoi(args, out):
    print(f"\n=== Tower of Hanoi  ({args.disks} disks) ===")
    prob = Hanoi(args.disks)
    res = search(prob, args.algo)
    report(res, f"| optimal = {2 ** args.disks - 1}")
    rec = hanoi_recursive(args.disks)
    print(f"   Recursive formula gives {len(rec)} moves (2^{args.disks} - 1); first moves: "
          + ", ".join(f"d{d}:{a}->{b}" for d, a, b in rec[:6]) + " ...")
    if args.disks <= 4:
        plot_hanoi_solution(prob, res, os.path.join(out, "hanoi_solution.png"))
    else:
        print("   (step-by-step picture skipped: more than 4 disks)")
    if args.disks <= 5:
        plot_hanoi_state_space(prob, res, os.path.join(out, "hanoi_state_space.png"))
    plot_comparison(prob, "Tower of Hanoi", os.path.join(out, "hanoi_compare.png"), trials=1000)
    if args.gif or args.show:
        n = args.disks
        animate_solution(res.states, res.moves, lambda ax, s: draw_hanoi(ax, s, n), f"Tower of Hanoi ({n} disks)",
                         os.path.join(out, "hanoi_solution.gif") if args.gif else None, args.show,
                         interval=700, figsize=(6, 3.8))
    return res


def main():
    ap = argparse.ArgumentParser(description="Two-One and Tower of Hanoi: search + visualization")
    ap.add_argument("--problem", choices=["twoone", "hanoi", "both"], default="both")
    ap.add_argument("--algo", choices=["bfs", "dfs"], default="bfs", help="search strategy used to solve")
    ap.add_argument("--coins", type=int, default=None, help="coins per side for Two-One (if omitted, you are asked)")
    ap.add_argument("--start", help="Two-One custom start state using 1, 2 and _  (e.g. 11_222)")
    ap.add_argument("--starts", nargs="+", help="solve several Two-One start states one after another")
    ap.add_argument("--sweep", type=int, metavar="N", help="solve Two-One for 1..N coins per side + growth chart")
    ap.add_argument("-i", "--interactive", action="store_true", help="interactive Two-One session (type sizes / start states)")
    ap.add_argument("--disks", type=int, default=3, help="number of Hanoi disks (default 3)")
    ap.add_argument("--outdir", default="search_output", help="folder for saved figures")
    ap.add_argument("--gif", action="store_true", help="also save animated GIFs")
    ap.add_argument("--show", action="store_true", help="open interactive windows")
    ap.add_argument("--no-show", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Two-One without an explicit size/start state -> ASK the user (works in any terminal or IDE).
    if (args.coins is None and not args.start and not args.starts and not args.sweep
            and args.problem in ("twoone", "both")):
        args.interactive = True
    if args.coins is None:
        args.coins = 2                                # default / suggested size

    if not args.show:
        matplotlib.use("Agg")                         # headless: just write files
    os.makedirs(args.outdir, exist_ok=True)

    if args.interactive and args.problem == "both":
        args.problem = "twoone"                       # interactive mode is for the Two-One problem
    if args.problem in ("twoone", "both"):
        run_two_one(args, args.outdir)
    if args.problem in ("hanoi", "both"):
        run_hanoi(args, args.outdir)

    print(f"\nFigures saved in: {os.path.abspath(args.outdir)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
