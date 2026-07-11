"""Small dependency-free Dinic max-flow implementation."""

from __future__ import annotations

from dataclasses import dataclass


INF = 10**9


@dataclass
class FlowEdge:
    to: int
    rev: int
    cap: float
    original: float


class Dinic:
    def __init__(self) -> None:
        self.graph: list[list[FlowEdge]] = []

    def node(self) -> int:
        self.graph.append([])
        return len(self.graph) - 1

    def add_edge(self, src: int, dst: int, cap: float) -> None:
        forward = FlowEdge(dst, len(self.graph[dst]), cap, cap)
        backward = FlowEdge(src, len(self.graph[src]), 0.0, 0.0)
        self.graph[src].append(forward)
        self.graph[dst].append(backward)

    def max_flow(self, source: int, sink: int) -> float:
        flow = 0.0
        while True:
            level = [-1] * len(self.graph)
            queue = [source]
            level[source] = 0
            for node in queue:
                for edge in self.graph[node]:
                    if edge.cap > 1e-9 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                return flow
            next_edge = [0] * len(self.graph)

            def dfs(node: int, pushed: float) -> float:
                if node == sink:
                    return pushed
                while next_edge[node] < len(self.graph[node]):
                    edge = self.graph[node][next_edge[node]]
                    if edge.cap > 1e-9 and level[node] + 1 == level[edge.to]:
                        sent = dfs(edge.to, min(pushed, edge.cap))
                        if sent > 1e-9:
                            edge.cap -= sent
                            self.graph[edge.to][edge.rev].cap += sent
                            return sent
                    next_edge[node] += 1
                return 0.0

            while True:
                pushed = dfs(source, INF)
                if pushed <= 1e-9:
                    break
                flow += pushed

    def reachable(self, source: int) -> set[int]:
        seen = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            for edge in self.graph[node]:
                if edge.cap > 1e-9 and edge.to not in seen:
                    seen.add(edge.to)
                    stack.append(edge.to)
        return seen
