"""Branch script graph: validation, traversal and waypoint route planning."""
from __future__ import annotations

from collections import deque
from typing import Any


# ---- validation -----------------------------------------------------------

def validate(content: dict[str, Any]) -> dict[str, list]:
    """Return validation markers for a script.

    Markers:
      cycles  : ids of nodes that lie on / lead into a directed cycle
      orphans : ids of nodes unreachable from the start node
      uncovered: node ids that have no options and no explicit terminal flag,
                 or an option whose target is missing/empty
    Missing-target options are additionally reported under `errors`.
    """
    nodes: list[dict] = content.get("nodes") or []
    start_id = content.get("start_node_id")
    node_ids = [n.get("id") for n in nodes]
    node_map = {n.get("id"): n for n in nodes}

    adjacency: dict[Any, list[Any]] = {nid: [] for nid in node_ids}
    dangling_options: list[dict] = []  # {"node_id", "option_id"}

    for node in nodes:
        for opt in node.get("options", []) or []:
            target = opt.get("target_id")
            if target and target in node_map:
                adjacency[node["id"]].append(target)
            else:
                dangling_options.append({"node_id": node["id"], "option_id": opt.get("id")})

    cycles = _find_cycle_nodes(node_ids, adjacency)

    orphans: list = []
    if start_id in node_map:
        reachable = _bfs_reachable(start_id, adjacency)
        orphans = [nid for nid in node_ids if nid not in reachable]
    else:
        # No valid start: every node is effectively orphaned.
        orphans = list(node_ids)

    uncovered = [d["node_id"] for d in dangling_options]
    errors = [
        {
            "type": "missing_target",
            "node_id": d["node_id"],
            "option_id": d["option_id"],
            "message": "选项未连接到任何节点",
        }
        for d in dangling_options
    ]
    if nodes and start_id not in node_map:
        errors.append({"type": "missing_start", "message": "起始节点不存在"})

    for node in nodes:
        options = node.get("options", []) or []
        if not options and not node.get("terminal"):
            uncovered.append(node["id"])

    return {
        "cycles": sorted(set(cycles)),
        "orphans": sorted(set(orphans)),
        "uncovered": sorted(set(uncovered)),
        "errors": errors,
        "ok": not cycles and not orphans and not errors and not uncovered,
    }


def _bfs_reachable(start: Any, adjacency: dict[Any, list]) -> set:
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _find_cycle_nodes(node_ids: list, adjacency: dict[Any, list]) -> list:
    """Nodes that belong to a directed cycle (Tarjan SCC, size>1 or self-loop)."""
    index_counter = [0]
    stack: list = []
    on_stack: set = set()
    indices: dict = {}
    lowlink: dict = {}
    cycle_nodes: set = set()

    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adjacency.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                cycle_nodes.update(component)
            elif v in adjacency.get(v, []):
                cycle_nodes.add(v)

    for nid in node_ids:
        if nid not in indices:
            strongconnect(nid)
    return list(cycle_nodes)


# ---- route planning -------------------------------------------------------

def plan_route(content: dict[str, Any], waypoints: list[dict]) -> dict[str, Any]:
    """Shortest choice sequence through every waypoint option at least once.

    Starts at ``start_node_id``; each step chooses one connected option and
    counts 1. A route is complete when every waypoint option has been chosen
    at least once AND the current node is terminal — either flagged
    ``terminal`` or simply having no options. Terminal-flagged nodes allow no
    further choices. Cycles and repeated visits are permitted: the search is
    a BFS over ``(node_id, collected-waypoints bitmask)`` states, so the first
    complete route found is shortest by choice count. When no route exists the
    result is ``solvable: False`` with a reason — never a partial path.
    """
    nodes: list[dict] = content.get("nodes") or []
    node_map = {n.get("id"): n for n in nodes}
    start_id = content.get("start_node_id")
    unsolvable = {"solvable": False, "choice_count": 0, "steps": []}
    if start_id not in node_map:
        return {**unsolvable, "reason": "起始节点未设置或不存在"}

    wp_index = {(w["node_id"], w["option_id"]): i for i, w in enumerate(waypoints)}
    full = (1 << len(waypoints)) - 1

    def is_end(node: dict) -> bool:
        return bool(node.get("terminal")) or not (node.get("options") or [])

    start_state = (start_id, 0)
    parent: dict[tuple, Any] = {start_state: None}  # state -> (prev_state, choice)
    queue = deque([start_state])
    goal = None
    while queue:
        cur_id, mask = queue.popleft()
        node = node_map[cur_id]
        if mask == full and is_end(node):
            goal = (cur_id, mask)
            break
        if node.get("terminal"):
            continue  # terminal nodes cannot continue
        for opt in node.get("options", []) or []:
            target = opt.get("target_id")
            if not target or target not in node_map:
                continue  # unconnected options are not selectable
            bit = wp_index.get((cur_id, opt.get("id")))
            nxt = (target, mask | (1 << bit) if bit is not None else mask)
            if nxt not in parent:
                parent[nxt] = ((cur_id, mask), (cur_id, opt.get("id"), target))
                queue.append(nxt)

    if goal is None:
        return {**unsolvable,
                "reason": "从起始节点出发，不存在经过全部勾选选项并到达终止节点（含无选项节点）的路线"}

    choices = []
    state = goal
    while parent[state] is not None:
        prev, choice = parent[state]
        choices.append(choice)
        state = prev
    choices.reverse()

    steps = []
    for node_id, option_id, target_id in choices:
        node = node_map[node_id]
        opt = next((o for o in node.get("options", []) if o.get("id") == option_id), {})
        target = node_map[target_id]
        steps.append({
            "node_id": node_id,
            "title": node.get("title", ""),
            "option_id": option_id,
            "option_label": opt.get("label", ""),
            "target_id": target_id,
            "target_title": target.get("title", ""),
            "waypoint": (node_id, option_id) in wp_index,
        })
    end_node = node_map[goal[0]]
    return {
        "solvable": True,
        "reason": None,
        "choice_count": len(steps),
        "steps": steps,
        "start_node_id": start_id,
        "start_title": node_map[start_id].get("title", ""),
        "end_node_id": goal[0],
        "end_title": end_node.get("title", ""),
        "end_terminal": bool(end_node.get("terminal")),
    }


# ---- revision impact analysis ---------------------------------------------
# Replay recorded answer sessions (editor preview runs AND share-answer runs)
# against a *candidate* script content (the current canvas, unsaved edits
# included) and report the FIRST point where each session diverges.
#
# A run's ``steps`` are stored as a node path: steps[i] is the node shown at
# position i; every step but the last records the option chosen there
# (steps[i].option_id) which led to steps[i+1]. Replay follows those historical
# choices in order on the candidate graph and checks the jump target at every
# step. Unfinished runs are only checked up to the part they actually answered.

IMPACT_TYPES = (
    "start_changed",        # candidate start node differs from the recorded one
    "node_missing",         # a node on the recorded path vanished from the canvas
    "option_missing",       # the recorded option vanished from its node
    "edge_retargeted",      # the option now points elsewhere / nowhere / to a ghost
    "early_terminal",       # the path now hits a terminal node before answers end
    "no_longer_finished",   # a finished run's end node is no longer a terminal
)


def _impact_breakpoint(rtype: str, reason: str, step_index: int,
                       node_id=None, node_title=None, node_exists=True,
                       option_id=None, option_label=None,
                       expected_target_id=None, actual_target_id=None,
                       current_start_id=None) -> dict[str, Any]:
    return {
        "type": rtype,
        "reason": reason,
        "step_index": step_index,
        "node_id": node_id,
        "node_title": node_title if node_title is not None else node_id,
        "node_exists": node_exists,
        "option_id": option_id,
        "option_label": option_label,
        "expected_target_id": expected_target_id,
        "actual_target_id": actual_target_id,
        "current_start_id": current_start_id,
    }


def _replay_run(node_map: dict, candidate_start, steps: list[dict],
                finished: bool) -> dict[str, Any]:
    """Replay one session; return status + first breakpoint (if any)."""
    def recorded_title(st: dict) -> str:
        return st.get("title") or st.get("node_id") or "(未知节点)"

    if not steps:
        # No recorded node at all — nothing can diverge.
        return {"status": "compatible", "breakpoint": None, "matched_choices": 0,
                "current_node_id": None}

    answered = sum(1 for st in steps[:-1] if st.get("option_id"))
    first = steps[0]
    start_rec = first.get("node_id")

    # 1) entry point check
    if start_rec not in node_map:
        return {"status": "affected", "matched_choices": 0, "current_node_id": start_rec,
                "breakpoint": _impact_breakpoint(
                    "node_missing",
                    f"起点节点「{recorded_title(first)}」已从画布消失，答卷无法从原入口开始",
                    0, node_id=start_rec, node_title=recorded_title(first), node_exists=False)}
    if candidate_start != start_rec:
        cur = node_map.get(candidate_start)
        cur_name = f"「{cur.get('title') or candidate_start}」" if cur else "未设置有效起始节点"
        return {"status": "affected", "matched_choices": 0, "current_node_id": start_rec,
                "breakpoint": _impact_breakpoint(
                    "start_changed",
                    f"起点已改变：原起点为「{recorded_title(first)}」，当前画布起点为{cur_name}",
                    0, node_id=start_rec, node_title=recorded_title(first),
                    current_start_id=candidate_start)}

    # 2) replay the historical choices in order, verifying every jump target
    matched = 0
    for i in range(len(steps) - 1):
        st, nxt = steps[i], steps[i + 1]
        nid, oid = st.get("node_id"), st.get("option_id")
        if not oid:
            break  # malformed tail: nothing more can be verified
        answer = st.get("answer") or oid
        expected_id = nxt.get("node_id")
        expected_title = recorded_title(nxt)
        node = node_map.get(nid)
        if node is None:
            return {"status": "affected", "matched_choices": matched,
                    "current_node_id": nid,
                    "breakpoint": _impact_breakpoint(
                        "node_missing",
                        f"第 {i + 1} 题节点「{recorded_title(st)}」已从画布消失，原路径在此中断",
                        i, node_id=nid, node_title=recorded_title(st), node_exists=False,
                        option_id=oid, option_label=answer)}
        if node.get("terminal"):
            return {"status": "affected", "matched_choices": matched,
                    "current_node_id": nid,
                    "breakpoint": _impact_breakpoint(
                        "early_terminal",
                        f"节点「{node.get('title') or nid}」已被标记为终止节点，"
                        f"原记录在此选择「{answer}」后仍有作答，现在会提前终止",
                        i, node_id=nid, node_title=node.get("title") or nid,
                        option_id=oid, option_label=answer,
                        expected_target_id=expected_id)}
        opt = next((o for o in (node.get("options") or []) if o.get("id") == oid), None)
        if opt is None:
            return {"status": "affected", "matched_choices": matched,
                    "current_node_id": nid,
                    "breakpoint": _impact_breakpoint(
                        "option_missing",
                        f"节点「{node.get('title') or nid}」上的原回答选项「{answer}」已消失",
                        i, node_id=nid, node_title=node.get("title") or nid,
                        option_id=oid, option_label=answer,
                        expected_target_id=expected_id)}
        target = opt.get("target_id")
        if not target:
            reason = (f"选项「{answer}」的跳转已被移除（当前未连接任何节点），"
                      f"原路径经此到达「{expected_title}」")
            rtype = "edge_retargeted"
            actual = None
        elif target not in node_map:
            reason = (f"选项「{answer}」现在指向的节点已从画布消失，"
                      f"原路径经此到达「{expected_title}」")
            rtype = "edge_retargeted"
            actual = target
        elif target != expected_id:
            actual_node = node_map[target]
            reason = (f"选项「{answer}」的跳转已改指：原来到「{expected_title}」，"
                      f"现在到「{actual_node.get('title') or target}」")
            rtype = "edge_retargeted"
            actual = target
        else:
            matched += 1
            continue
        return {"status": "affected", "matched_choices": matched, "current_node_id": nid,
                "breakpoint": _impact_breakpoint(
                    rtype, reason, i, node_id=nid, node_title=node.get("title") or nid,
                    option_id=oid, option_label=answer,
                    expected_target_id=expected_id, actual_target_id=actual)}

    # 3) all recorded choices replay identically — check the end condition
    cur_id = steps[-1].get("node_id")
    cur_node = node_map.get(cur_id)
    if cur_node is None:
        # Defensive: reached node was always validated as an existing target above.
        return {"status": "affected", "matched_choices": matched,
                "current_node_id": cur_id,
                "breakpoint": _impact_breakpoint(
                    "node_missing", f"当前节点「{cur_id}」已从画布消失",
                    len(steps) - 1, node_id=cur_id, node_title=cur_id, node_exists=False)}
    ends_now = bool(cur_node.get("terminal")) or not (cur_node.get("options") or [])
    idx = len(steps) - 1
    if finished and not ends_now:
        return {"status": "affected", "matched_choices": matched, "current_node_id": cur_id,
                "breakpoint": _impact_breakpoint(
                    "no_longer_finished",
                    f"原答卷在「{cur_node.get('title') or cur_id}」结束，"
                    f"但该节点现在不再是终止节点，答卷不会在此结束",
                    idx, node_id=cur_id, node_title=cur_node.get("title") or cur_id)}
    if not finished and ends_now:
        return {"status": "affected", "matched_choices": matched, "current_node_id": cur_id,
                "breakpoint": _impact_breakpoint(
                    "early_terminal",
                    f"答卷尚未答完，但当前所在节点「{cur_node.get('title') or cur_id}」"
                    f"现在是终止节点，继续作答会在此提前终止",
                    idx, node_id=cur_id, node_title=cur_node.get("title") or cur_id)}

    # every recorded step (and, for finished runs, the end condition) still matches
    return {"status": "compatible", "breakpoint": None,
            "matched_choices": matched, "current_node_id": cur_id}


def analyze_impact(content: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay every recorded session of one script against candidate content.

    ``runs`` items: {run_id, mode ('script'|'share'), finished, created_at, steps}.
    Pure read-only computation: nothing here writes to the database, modifies
    run records or touches coverage statistics.
    """
    node_map = {n.get("id"): n for n in content.get("nodes", [])}
    candidate_start = content.get("start_node_id")

    sessions: list[dict[str, Any]] = []
    for r in runs:
        steps = r.get("steps") or []
        result = _replay_run(node_map, candidate_start, steps, bool(r.get("finished")))
        sessions.append({
            "run_id": r.get("run_id"),
            "mode": r.get("mode"),
            "finished": bool(r.get("finished")),
            "created_at": r.get("created_at"),
            "step_count": len(steps),
            "answered_count": sum(1 for st in steps[:-1] if st.get("option_id")),
            "steps": steps,
            **result,
        })

    # affected sessions first (newest first), compatible after
    sessions.sort(key=lambda s: (s["status"] != "affected", -(s["run_id"] or 0)))

    reason_counts: dict[str, int] = {}
    affected = [s for s in sessions if s["status"] == "affected"]
    for s in affected:
        reason_counts[s["breakpoint"]["type"]] = reason_counts.get(s["breakpoint"]["type"], 0) + 1

    return {
        "has_runs": bool(sessions),
        "run_count": len(sessions),
        "compatible_count": len(sessions) - len(affected),
        "affected_count": len(affected),
        "preview_runs": sum(1 for s in sessions if s["mode"] == "script"),
        "share_runs": sum(1 for s in sessions if s["mode"] == "share"),
        "reason_counts": reason_counts,
        "sessions": sessions,
    }


# ---- runtime traversal ----------------------------------------------------

def step(content: dict, node_id: str, option_id: str | None = None) -> dict[str, Any]:
    """Resolve the next node from `node_id` via `option_id`.

    option_id None means "start": resolve from start_node_id.
    Returns {node, finished} or raises ValueError.
    """
    node_map = {n["id"]: n for n in content.get("nodes", [])}
    if option_id is None:
        start = content.get("start_node_id")
        if start not in node_map:
            raise ValueError("起始节点不存在")
        return {"node": node_map[start], "finished": False}

    current = node_map.get(node_id)
    if current is None:
        raise ValueError("当前节点不存在")
    opt = next((o for o in current.get("options", []) if o.get("id") == option_id), None)
    if opt is None:
        raise ValueError("该选项不存在")
    target = opt.get("target_id")
    if not target or target not in node_map:
        raise ValueError("选项未连接到目标节点")
    return {"node": node_map[target], "finished": False}
