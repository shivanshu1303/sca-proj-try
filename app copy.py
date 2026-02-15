import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from collections import defaultdict
import heapq
import hashlib

try:
    from streamlit_drawable_canvas import st_canvas
except Exception as e:
    st.error(
        "Missing dependency: streamlit-drawable-canvas. "
        "Run: pip install streamlit-drawable-canvas"
    )
    raise

# -----------------------------
# Helpers
# -----------------------------

# DEFAULT_EDGE_COLUMNS = ["source", "target", "cost", "time", "co2", "reliability"]
DEFAULT_EDGE_COLUMNS = ["source", "target", "weight"]
DEFAULT_NODE_COLUMNS = ["node_id", "label", "type", "x", "y"]

MAX_CANVAS_W = 1100
MAX_CANVAS_H = 700
MAX_PIXELS   = 25_000_000  # safety guard (~25MP)

def load_and_resize_image(file_bytes: bytes, max_w: int, max_h: int):
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")

    # Safety: prevent huge decompression/memory bombs
    w, h = img.size
    if w * h > MAX_PIXELS:
        # force shrink early (keeps aspect ratio)
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    # Final: resize to canvas-friendly size
    img.thumbnail((max_w, max_h), Image.LANCZOS)  # in-place, keeps aspect ratio
    return img

@dataclass
class Node:
    node_id: str
    label: str
    ntype: str
    x: float
    y: float

def _safe_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default

def nearest_node_id(nodes: List[Node], x: float, y: float) -> Optional[str]:
    """Return node_id of the closest node, or None if nodes is empty."""
    if not nodes:
        return None
    best_id, best_d = None, float("inf")
    for n in nodes:
        d = (n.x - x) ** 2 + (n.y - y) ** 2
        if d < best_d:
            best_d = d
            best_id = n.node_id
    return best_id

def parse_canvas_objects(canvas_json: Dict, node_snap_px: float = 25.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse circles (nodes) and lines (edges) from drawable canvas JSON.
    Returns (nodes_df, edges_df).
    - Nodes come from objects with type == "circle"
    - Edges come from objects with type == "line"
    We map line endpoints to nearest circles (node centers) if within node_snap_px.
    """
    objs = (canvas_json or {}).get("objects", []) or []

    # Collect circles as nodes
    nodes: List[Node] = []
    circle_objs = [o for o in objs if o.get("type") == "circle"]

    for idx, o in enumerate(circle_objs, start=1):
        # FabricJS circles have left/top with radius; center approx = left+radius, top+radius
        left = _safe_float(o.get("left"))
        top = _safe_float(o.get("top"))
        radius = _safe_float(o.get("radius"), default=0.0)
        x = left + radius
        y = top + radius

        node_id = f"N{idx}"
        nodes.append(Node(node_id=node_id, label=node_id, ntype="Node", x=x, y=y))

    nodes_df = pd.DataFrame(
        [{
            "node_id": n.node_id,
            "label": n.label,
            "type": n.ntype,
            "x": round(n.x, 2),
            "y": round(n.y, 2)
        } for n in nodes],
        columns=DEFAULT_NODE_COLUMNS
    )

    # # Collect line edges
    line_objs = [o for o in objs if o.get("type") == "line"]

    edges_rows = []
    for o in line_objs:
        # # FabricJS stores endpoints in x1,y1,x2,y2 relative to object; absolute endpoints are trickier.
        # # But st_canvas returns absolute x1,y1,x2,y2 for lines drawn in most cases.
        # x1 = _safe_float(o.get("x1"))
        # y1 = _safe_float(o.get("y1"))
        # x2 = _safe_float(o.get("x2"))
        # y2 = _safe_float(o.get("y2"))

        # if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
        #     continue
        
        pts = line_abs_endpoints(o)
        if pts is None:
            continue
        x1, y1, x2, y2 = pts

        s_id = nearest_node_id(nodes, x1, y1)
        t_id = nearest_node_id(nodes, x2, y2)

        # Snap only if close enough to a node center
        def within_snap(nid, x, y):
            if nid is None:
                return False
            n = next((nn for nn in nodes if nn.node_id == nid), None)
            if n is None:
                return False
            return math.hypot(n.x - x, n.y - y) <= node_snap_px

        if not within_snap(s_id, x1, y1) or not within_snap(t_id, x2, y2):
            # If it doesn't snap, we skip: user can still add it manually in the table.
            continue

        # edges_rows.append({
        #     "source": s_id,
        #     "target": t_id,
        #     "cost": 0.0,
        #     "time": 0.0,
        #     "co2": 0.0,
        #     "reliability": 0.95,
        # })
        edges_rows.append({
            "source": s_id,
            "target": t_id,
            "weight": 1.0,
        })

    edges_df = pd.DataFrame(edges_rows, columns=DEFAULT_EDGE_COLUMNS)

    return nodes_df, edges_df

def line_abs_endpoints(o: dict) -> tuple[float, float, float, float] | None:
    """
    Try to return absolute (canvas) endpoints for a Fabric.js 'line' object.
    Many Fabric JSONs store x1..y2 relative to the object's origin and use left/top for placement.
    """
    x1 = _safe_float(o.get("x1"))
    y1 = _safe_float(o.get("y1"))
    x2 = _safe_float(o.get("x2"))
    y2 = _safe_float(o.get("y2"))

    left = _safe_float(o.get("left"), default=0.0)
    top  = _safe_float(o.get("top"),  default=0.0)

    if any(np.isnan(v) for v in [x1, y1, x2, y2]):
        return None

    # Most common fix: endpoints are relative → shift by left/top
    ax1, ay1, ax2, ay2 = left + x1, top + y1, left + x2, top + y2

    return ax1, ay1, ax2, ay2

# def validate_edges(edges_df: pd.DataFrame, directed: bool) -> List[str]:
#     issues = []
#     if edges_df.empty:
#         issues.append("No edges found/entered yet.")

#     # Required columns
#     missing = [c for c in DEFAULT_EDGE_COLUMNS if c not in edges_df.columns]
#     if missing:
#         issues.append(f"Edge table missing columns: {missing}")

#     # Basic type checks
#     for c in ["cost", "time", "co2", "reliability"]:
#         if c in edges_df.columns:
#             edges_df[c] = pd.to_numeric(edges_df[c], errors="coerce")

#     if "reliability" in edges_df.columns:
#         bad = edges_df[(edges_df["reliability"] < 0) | (edges_df["reliability"] > 1)]
#         if len(bad) > 0:
#             issues.append("Reliability must be between 0 and 1 for all edges.")

#     # If undirected, ensure (u,v) duplicates are handled later; not an error.
#     return issues

def validate_edges(edges_df: pd.DataFrame, directed: bool) -> list[str]:
    issues = []
    if edges_df.empty:
        issues.append("No edges found/entered yet.")
        return issues

    missing = [c for c in DEFAULT_EDGE_COLUMNS if c not in edges_df.columns]
    if missing:
        issues.append(f"Edge table missing columns: {missing}")
        return issues

    # Numeric check
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce")
    if edges_df["weight"].isna().any():
        issues.append("Some edges have non-numeric / blank weights. Fill all weights.")

    # Non-negative check (recommended for Dijkstra later)
    if (edges_df["weight"] < 0).any():
        issues.append("Some edges have negative weights. Dijkstra won't be valid on this weight.")

    # Basic endpoint check
    if edges_df["source"].isna().any() or edges_df["target"].isna().any():
        issues.append("Some edges have blank source/target.")

    return issues

def export_bundle(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, meta: Dict) -> bytes:
    payload = {
        "meta": meta,
        "nodes": nodes_df.to_dict(orient="records"),
        "edges": edges_df.to_dict(orient="records")
    }
    return json.dumps(payload, indent=2).encode("utf-8")

def build_adj_list(edges: pd.DataFrame, directed: bool) -> dict[str, list[tuple[str, float]]]:
    """Adjacency list: u -> [(v, w), ...]"""
    adj = defaultdict(list)
    for _, r in edges.iterrows():
        u = str(r["source"]).strip()
        v = str(r["target"]).strip()
        w = float(r["weight"])
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return dict(adj)

def dijkstra_steps(adj: dict[str, list[tuple[str, float]]], nodes: list[str], source: str):
    """
    Returns:
    dist: dict[node] -> shortest distance from source
    prev: dict[node] -> predecessor on shortest path tree
    steps_df: DataFrame with iteration snapshots
    """
    INF = float("inf")
    dist = {n: INF for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0.0

    visited = set()
    pq = [(0.0, source)]

    step_rows = []
    it = 0

    while pq:
        d_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        # Snapshot BEFORE relaxing outgoing edges (typical teaching style)
        row = {"iter": it, "picked": u, "picked_dist": d_u}
        for n in nodes:
            row[f"dist[{n}]"] = dist[n]
            row[f"prev[{n}]"] = prev[n] if prev[n] is not None else ""
        step_rows.append(row)

        # Relax neighbors
        for v, w in adj.get(u, []):
            if v in visited:
                continue
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

        it += 1

    steps_df = pd.DataFrame(step_rows)
    return dist, prev, steps_df

def reconstruct_path(prev: dict[str, str | None], source: str, target: str):
    if source == target:
        return [source]
    if prev.get(target) is None:
        return []  # no path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev.get(cur)
    path.reverse()
    if path and path[0] == source:
        return path
    return []

def build_edge_list(edges: pd.DataFrame, directed: bool) -> list[tuple[str, str, float]]:
    """Return list of (u, v, w). If undirected, adds both (u,v) and (v,u)."""
    eds = []
    for _, r in edges.iterrows():
        u = str(r["source"]).strip()
        v = str(r["target"]).strip()
        w = float(r["weight"])
        eds.append((u, v, w))
        if not directed:
            eds.append((v, u, w))
    return eds

def bellman_ford_steps(nodes: list[str], edges_list: list[tuple[str, str, float]], source: str):
    """
    Returns:
    dist, prev, steps_df, has_negative_cycle, neg_cycle_edges
    steps_df logs (iteration, relaxations_count, dist/prev snapshot).
    """
    INF = float("inf")
    dist = {n: INF for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0.0

    step_rows = []
    V = len(nodes)

    # Relax edges V-1 times
    for it in range(1, V):
        changed = 0
        for (u, v, w) in edges_list:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                changed += 1

        # snapshot
        row = {"iter": it, "relaxations": changed}
        for n in nodes:
            row[f"dist[{n}]"] = dist[n]
            row[f"prev[{n}]"] = prev[n] if prev[n] is not None else ""
        step_rows.append(row)

        # Early stop if no changes
        if changed == 0:
            break

    # Detect negative cycle (one more pass)
    neg_cycle_edges = []
    for (u, v, w) in edges_list:
        if dist[u] != INF and dist[u] + w < dist[v]:
            neg_cycle_edges.append((u, v, w))

    has_neg_cycle = len(neg_cycle_edges) > 0
    steps_df = pd.DataFrame(step_rows)
    return dist, prev, steps_df, has_neg_cycle, neg_cycle_edges

def floyd_warshall(nodes: list[str], edges_list: list[tuple[str, str, float]], keep_steps: bool = False):
    """
    Returns:
    dist_df, next_df, steps (optional list of snapshots)
    next_df helps reconstruct paths.
    """
    INF = float("inf")
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    dist = [[INF]*n for _ in range(n)]
    nxt  = [[None]*n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0.0
        nxt[i][i] = nodes[i]

    for (u, v, w) in edges_list:
        i, j = idx[u], idx[v]
        if w < dist[i][j]:   # keep best direct edge
            dist[i][j] = w
            nxt[i][j] = v

    steps = []
    for k in range(n):
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            for j in range(n):
                dkj = dist[k][j]
                if dkj == INF:
                    continue
                nd = dik + dkj
                if nd < dist[i][j]:
                    dist[i][j] = nd
                    nxt[i][j] = nxt[i][k]
        if keep_steps:
            # snapshot after each k
            steps.append(pd.DataFrame(dist, index=nodes, columns=nodes))

    dist_df = pd.DataFrame(dist, index=nodes, columns=nodes)
    next_df = pd.DataFrame(nxt, index=nodes, columns=nodes)
    return dist_df, next_df, steps

def fw_reconstruct_path(next_df: pd.DataFrame, source: str, target: str) -> list[str]:
    if pd.isna(next_df.loc[source, target]) or next_df.loc[source, target] is None:
        return []
    path = [source]
    while source != target:
        source = next_df.loc[source, target]
        if source is None or (isinstance(source, float) and np.isnan(source)):
            return []
        path.append(source)
        # safety
        if len(path) > len(next_df.index) + 5:
            return []
    return path

class DSU:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def kruskal_mst_steps(nodes: list[str], edges_df: pd.DataFrame):
    """
    edges_df contains (source, target, weight).
    Treat as undirected: we use unique undirected edges (min weight if duplicated).
    Returns: mst_edges_df, total_weight, steps_df
    """
    # Build unique undirected edge set
    tmp = edges_df.copy()
    tmp["u"] = tmp["source"].astype(str)
    tmp["v"] = tmp["target"].astype(str)
    tmp["a"] = tmp[["u","v"]].min(axis=1)
    tmp["b"] = tmp[["u","v"]].max(axis=1)
    tmp["w"] = pd.to_numeric(tmp["weight"], errors="coerce")

    und = tmp.groupby(["a","b"], as_index=False)["w"].min()
    und = und.sort_values("w", kind="mergesort").reset_index(drop=True)

    dsu = DSU(nodes)
    chosen = []
    step_rows = []
    total = 0.0
    step = 0

    for _, r in und.iterrows():
        a, b, w = r["a"], r["b"], float(r["w"])
        merged = dsu.union(a, b)

        step_rows.append({
            "step": step,
            "edge": f"{a}-{b}",
            "weight": w,
            "action": "TAKE" if merged else "SKIP (cycle)",
            "mst_weight_so_far": total + (w if merged else 0.0),
            "mst_edges_so_far": len(chosen) + (1 if merged else 0),
        })

        if merged:
            chosen.append((a, b, w))
            total += w
            if len(chosen) == len(nodes) - 1:
                break
        step += 1

    mst_edges_df = pd.DataFrame(chosen, columns=["u", "v", "weight"])
    steps_df = pd.DataFrame(step_rows)
    return mst_edges_df, total, steps_df

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Supply Chain Graph Input Studio", layout="wide")

st.title("Supply Chain Graph Input Studio (Input Only)")
st.caption(
    "Upload a hand-drawn network image (assist-trace), or enter edges manually/CSV. "
    "This step produces a clean node list + edge list for later algorithms."
)

with st.sidebar:
    st.header("Graph Settings")
    graph_name = st.text_input("Graph name", value="MyNetwork")
    directed = st.toggle("Directed graph?", value=False)
    st.markdown("---")
    st.subheader("Interpretation (MBA layer)")
    st.write("Select which lane attributes you will later optimize.")
    objective_options = st.multiselect(
        "Lane attributes captured",
        options=["cost", "time", "co2", "reliability"],
        default=["cost", "time", "co2", "reliability"]
    )
    st.markdown("---")
    snap_px = st.slider("Snap tolerance (px) for line endpoints → node", 5, 80, 75)
    st.session_state.graph_name = graph_name
    st.session_state.directed = directed

tab1, tab2, tab3 = st.tabs(["1) Upload image + trace", "2) Upload CSV edge list", "3) Manual entry"])

# Persistent state
if "nodes_df" not in st.session_state:
    st.session_state.nodes_df = pd.DataFrame(columns=DEFAULT_NODE_COLUMNS)
if "edges_df" not in st.session_state:
    st.session_state.edges_df = pd.DataFrame(columns=DEFAULT_EDGE_COLUMNS)

# -------- Tab 1: Image + Trace
with tab1:
    st.subheader("Upload image and trace nodes/edges (assisted)")
    img_file = st.file_uploader("Upload a graph image (png/jpg)", type=["png", "jpg", "jpeg"])  # :contentReference[oaicite:3]{index=3}

    # bg_image = None
    # if img_file is not None:
    #     bg_image = Image.open(img_file).convert("RGB")
    #     st.image(bg_image, caption="Reference image (you will trace on the canvas below)", use_container_width=True)
    
    bg_image = None
    canvas_key = "canvas_empty"
    
    if img_file is not None:
        file_bytes = img_file.getvalue()
        
        img_hash = hashlib.md5(file_bytes).hexdigest()[:10]
        canvas_key = f"canvas_{img_hash}"
        
        bg_image = load_and_resize_image(file_bytes, MAX_CANVAS_W, MAX_CANVAS_H)
        st.image(bg_image, caption="Reference image (resized for canvas)", use_container_width=True)

    st.markdown("### Trace canvas")
    st.info(
        "How to use:\n"
        "1) Select **circle** tool and place circles on nodes.\n"
        "2) Select **line** tool and draw edges between node circles.\n"
        "3) Click **Parse traced objects** to populate node/edge tables.\n"
        "Then go to the Manual tab to label nodes and fill lane attributes."
    )

    drawing_mode = st.selectbox("Drawing tool", ["circle", "line", "transform"])
    stroke_width = st.slider("Stroke width", 1, 8, 3)
    radius = st.slider("Circle radius", 6, 30, 12)

    if bg_image is not None:
        canvas_res = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="rgba(255, 0, 0, 0.9)",
            background_image=bg_image.copy(),
            update_streamlit=True,
            height=int(bg_image.height),
            width=int(bg_image.width),
            drawing_mode=drawing_mode,
            key=canvas_key,
            initial_drawing=None,
            display_toolbar=True,
            # circle settings via JSON aren't directly parameterized; radius from object parsing
        )  # :contentReference[oaicite:4]{index=4}

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Parse traced objects → node/edge tables"):
                if canvas_res.json_data is None:
                    st.warning("Nothing to parse yet.")
                else:
                    nodes_df, edges_df = parse_canvas_objects(canvas_res.json_data, node_snap_px=float(snap_px))
                    # Merge into session state (overwrite for now)
                    st.session_state.nodes_df = nodes_df
                    # Append edges found
                    st.session_state.edges_df = edges_df if not edges_df.empty else st.session_state.edges_df
                    st.success(f"Parsed {len(nodes_df)} nodes and {len(edges_df)} edges from the canvas.")
        with colB:
            st.write("Parsed preview:")
            st.dataframe(st.session_state.nodes_df, use_container_width=True)
            st.dataframe(st.session_state.edges_df, use_container_width=True)

    else:
        st.warning("Upload an image first to enable tracing.")

# -------- Tab 2: CSV edge list upload
with tab2:
    st.subheader("Upload CSV edge list (guaranteed input path)")
    # st.write("Expected columns: source,target,cost,time,co2,reliability (you can omit some; we’ll add defaults).")
    st.write("Expected columns: source,target,weight")

    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")  # :contentReference[oaicite:5]{index=5}
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        # Ensure columns exist
        for c in DEFAULT_EDGE_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        # Default reliability if missing
        # df["reliability"] = df["reliability"].fillna(0.95)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)

        st.session_state.edges_df = df[DEFAULT_EDGE_COLUMNS].copy()

        # Create nodes from unique endpoints
        nodes = sorted(set(df["source"].astype(str)).union(set(df["target"].astype(str))))
        nodes_df = pd.DataFrame([{
            "node_id": n,
            "label": n,
            "type": "Node",
            "x": np.nan,
            "y": np.nan
        } for n in nodes], columns=DEFAULT_NODE_COLUMNS)

        st.session_state.nodes_df = nodes_df
        st.success(f"Loaded {len(df)} edges and {len(nodes)} nodes from CSV.")

    st.dataframe(st.session_state.edges_df, use_container_width=True)

# -------- Tab 3: Manual entry
# with tab3:
#     st.subheader("Manual entry (edit tables directly)")
#     st.write("Edit nodes and edges. This is also where you set lane attributes (cost/time/CO₂/reliability).")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("### Nodes")
#         nodes_df = st.session_state.nodes_df.copy()
#         nodes_df = st.data_editor(  # :contentReference[oaicite:6]{index=6}
#             nodes_df,
#             num_rows="dynamic",
#             use_container_width=True,
#             hide_index=True
#         )
#         st.session_state.nodes_df = nodes_df

#     with col2:
#         st.markdown("### Edges")
#         edges_df = st.session_state.edges_df.copy()
#         if "weight" not in edges_df.columns:
#             edges_df["weight"] = 1.0
#         edges_df = st.data_editor(  # :contentReference[oaicite:7]{index=7}
#             edges_df,
#             num_rows="dynamic",
#             use_container_width=True,
#             hide_index=True,
#             column_config={
#                 "weight": st.column_config.NumberColumn(
#                     "Weight",    
#                     help="Scalar edge weight used by algorithms (negative allowed for Bellman–Ford / Floyd–Warshall).",
#                     # help="Scalar edge weight used by algorithms (for now).",
#                     min_value=None,   # recommended
#                     step=1.0,
#                     required=True,
#                 ),
#                 "source": st.column_config.TextColumn("Source", required=True),
#                 "target": st.column_config.TextColumn("Target", required=True),
#             },
#         )
#         st.session_state.edges_df = edges_df

# -------- Tab 3: Manual entry
with tab3:
    st.subheader("Manual entry (edit tables directly)")
    st.write("Edit nodes and edges, then click **Save changes**.")

    col1, col2 = st.columns(2)

    # ---- NODES FORM
    with col1:
        st.markdown("### Nodes")

        with st.form("nodes_form", clear_on_submit=False):
            edited_nodes = st.data_editor(
                st.session_state.nodes_df,
                key="nodes_editor",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
            )
            save_nodes = st.form_submit_button("Save node changes")

        if save_nodes:
            st.session_state.nodes_df = edited_nodes

    # ---- EDGES FORM
    with col2:
        st.markdown("### Edges")

        # Ensure column exists ONCE (don’t keep resetting)
        if "weight" not in st.session_state.edges_df.columns:
            st.session_state.edges_df["weight"] = 1.0

        with st.form("edges_form", clear_on_submit=False):
            edited_edges = st.data_editor(
                st.session_state.edges_df,
                key="edges_editor",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "weight": st.column_config.NumberColumn(
                        "Weight",
                        help="Negative allowed for Bellman–Ford / Floyd–Warshall.",
                        min_value=None,
                        step=1.0,
                        required=True,
                    ),
                    "source": st.column_config.TextColumn("Source", required=True),
                    "target": st.column_config.TextColumn("Target", required=True),
                },
            )
            save_edges = st.form_submit_button("Save edge changes")

        if save_edges:
            st.session_state.edges_df = edited_edges


# -------- Validation + Export
# st.markdown("---")
# st.subheader("Validate & Export")

# nodes_df = st.session_state.nodes_df.copy()
# edges_df = st.session_state.edges_df.copy()

# issues = validate_edges(edges_df, directed=directed)
# if issues:
#     st.warning("Validation notes:")
#     for it in issues:
#         st.write(f"- {it}")
# else:
#     st.success("Basic validation passed.")

# # Extra: show whether negative weights exist (matters later for algorithm choice)
# if "cost" in edges_df.columns and pd.to_numeric(edges_df["cost"], errors="coerce").min(skipna=True) < 0:
#     st.info("Note: Negative edge costs detected. Later, Dijkstra will be disabled for that weight; Bellman–Ford/Floyd–Warshall will apply.")

# meta = {
#     "graph_name": graph_name,
#     "directed": directed,
#     "captured_attributes": objective_options
# }

# bundle_bytes = export_bundle(nodes_df, edges_df, meta=meta)
# st.download_button(
#     "Download graph as JSON (nodes + edges + meta)",
#     data=bundle_bytes,
#     file_name=f"{graph_name}.json",
#     mime="application/json"
# )

# # Also provide CSV downloads
# st.download_button(
#     "Download edges.csv",
#     data=edges_df.to_csv(index=False).encode("utf-8"),
#     file_name=f"{graph_name}_edges.csv",
#     mime="text/csv"
# )
# st.download_button(
#     "Download nodes.csv",
#     data=nodes_df.to_csv(index=False).encode("utf-8"),
#     file_name=f"{graph_name}_nodes.csv",
#     mime="text/csv"
# )

# st.caption(
#     "Later, we’ll convert this edge list into a graph object (e.g., NetworkX from a pandas edge list) "
#     "and run algorithms + step tables."  # :contentReference[oaicite:8]{index=8}
# )

st.markdown("---")
st.subheader("Finish input (clean → validate → export)")

finalize = st.button("Finalize input (clean + validate + enable algorithms)")

if finalize:
    nodes_df = st.session_state.nodes_df.copy()
    edges_df = st.session_state.edges_df.copy()

    # ---------- 1) CLEAN / NORMALIZE ----------
    # Ensure required columns exist
    for col in ["node_id"]:
        if col not in nodes_df.columns:
            nodes_df[col] = ""

    for col in ["source", "target", "weight"]:
        if col not in edges_df.columns:
            edges_df[col] = np.nan

    # Force types (important: data_editor can create mixed types; keep columns editable & stable)
    nodes_df["node_id"] = nodes_df["node_id"].astype(str).str.strip()
    edges_df["source"] = edges_df["source"].astype(str).str.strip()
    edges_df["target"] = edges_df["target"].astype(str).str.strip()
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce")

    # Drop blank rows (common after dynamic editing)
    nodes_df = nodes_df[nodes_df["node_id"].ne("")].copy()
    edges_df = edges_df[(edges_df["source"].ne("")) & (edges_df["target"].ne(""))].copy()

    # Optional: remove self-loops (you can allow them, but many supply chain graphs won't have them)
    remove_self_loops = st.checkbox("Remove self-loops (source == target)", value=True)
    if remove_self_loops:
        edges_df = edges_df[edges_df["source"] != edges_df["target"]].copy()

    # Optional: handle duplicate edges
    # For a simple Graph later, duplicates get overwritten; we aggregate to one edge here.
    dedupe_mode = st.selectbox(
        "If duplicate edges exist (same source,target), keep…",
        ["Minimum weight", "Maximum weight", "Average weight", "Keep first"],
        index=0,
    )
    if not edges_df.empty:
        if dedupe_mode == "Minimum weight":
            edges_df = edges_df.groupby(["source", "target"], as_index=False)["weight"].min()
        elif dedupe_mode == "Maximum weight":
            edges_df = edges_df.groupby(["source", "target"], as_index=False)["weight"].max()
        elif dedupe_mode == "Average weight":
            edges_df = edges_df.groupby(["source", "target"], as_index=False)["weight"].mean()
        else:
            edges_df = edges_df.drop_duplicates(subset=["source", "target"], keep="first")

    # Keep only the ideal columns (everything else stays out of algorithm engine by default)
    nodes_keep = [c for c in ["node_id", "label", "type", "x", "y"] if c in nodes_df.columns]
    edges_keep = ["source", "target", "weight"]
    nodes_clean = nodes_df[nodes_keep].copy()
    edges_clean = edges_df[edges_keep].copy()

    # ---------- 2) VALIDATE ----------
    issues = []

    # Nodes unique
    if nodes_clean["node_id"].duplicated().any():
        dups = nodes_clean.loc[nodes_clean["node_id"].duplicated(), "node_id"].unique().tolist()
        issues.append(f"Duplicate node_id(s): {dups}")

    # Edges weight present + numeric
    if edges_clean["weight"].isna().any():
        issues.append("Some edges have blank / non-numeric weights. Fill them in the Manual Entry tab.")

    # Edges must reference valid nodes
    node_set = set(nodes_clean["node_id"].tolist())
    bad_src = edges_clean[~edges_clean["source"].isin(node_set)]
    bad_tgt = edges_clean[~edges_clean["target"].isin(node_set)]
    if len(bad_src) > 0 or len(bad_tgt) > 0:
        issues.append("Some edges reference node_id(s) that do not exist in the Nodes table.")

    # Weight sign (important for Dijkstra later)
    if (edges_clean["weight"] < 0).any():
        issues.append("Negative weights detected. Later: Dijkstra will be invalid on this weight (use Bellman–Ford/Floyd–Warshall).")

    if issues:
        st.error("Fix these before moving to algorithms:")
        for it in issues:
            st.write(f"- {it}")
    else:
        st.success("Input is clean and algorithm-ready ✅")

    # ---------- 3) EXPORT ----------
    # Streamlit recommends converting to downloadable bytes and using download_button. :contentReference[oaicite:2]{index=2}

    meta = {
        "graph_name": st.session_state.get("graph_name", "MyNetwork"),
        "directed": st.session_state.get("directed", True),
        "notes": "Scalar-weight graph input for shortest-path / MST algorithms."
    }

    graph_json = json.dumps(
        {"meta": meta, "nodes": nodes_clean.to_dict("records"), "edges": edges_clean.to_dict("records")},
        indent=2
    ).encode("utf-8")

    colA, colB, colC = st.columns(3)
    with colA:
        st.download_button(
            "Download nodes.csv",
            data=nodes_clean.to_csv(index=False).encode("utf-8"),
            file_name="nodes.csv",
            mime="text/csv"
        )
    with colB:
        st.download_button(
            "Download edges.csv",
            data=edges_clean.to_csv(index=False).encode("utf-8"),
            file_name="edges.csv",
            mime="text/csv"
        )
    with colC:
        st.download_button(
            "Download graph.json",
            data=graph_json,
            file_name="graph.json",
            mime="application/json"
        )

    # Save back the clean versions for the next phase (algorithms)
    st.session_state.nodes_df_clean = nodes_clean
    st.session_state.edges_df_clean = edges_clean

else:
    st.info("Edit nodes/edges above, then click Finalize.")

st.markdown("---")
st.header("Algorithm Studio")
st.caption(f"Building graph as {'DIRECTED' if directed else 'UNDIRECTED'} (undirected adds each edge both ways).")

if "nodes_df_clean" not in st.session_state or "edges_df_clean" not in st.session_state:
    st.info("First finish the input section (clean → validate → export).")
else:
    nodes_clean = st.session_state.nodes_df_clean.copy()
    edges_clean = st.session_state.edges_df_clean.copy()

    # Basic prerequisites
    nodes_list = nodes_clean["node_id"].astype(str).tolist()

    # Build adjacency list
    directed = st.session_state.get("directed", True)
    adj = build_adj_list(edges_clean, directed=directed)

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        # algo = st.selectbox("Algorithm", ["Dijkstra (single-source shortest path)"])
        algo = st.selectbox(
            "Algorithm",
            [
                "Dijkstra (single-source shortest path)",
                "Bellman–Ford (handles negative edges)",
                "Floyd–Warshall (all-pairs shortest paths)",
                "MST (Kruskal) - undirected only",
            ],
            key="algo_choice"
        )

    # with col2:
    #     source = st.selectbox("Source node", nodes_list)
    # with col3:
    #     target = st.selectbox("Target node (optional)", ["(none)"] + nodes_list)
    
    needs_source_target = not algo.startswith("MST")
    needs_target = algo.startswith("Dijkstra") or algo.startswith("Bellman") or algo.startswith("Floyd")

    source = None
    target = "(none)"

    if needs_source_target:
        with col2:
            source = st.selectbox("Source node", nodes_list, key="source_node")
        with col3:
            target = st.selectbox("Target node (optional)", ["(none)"] + nodes_list, key="target_node")
    else:
        # Use the space for something useful instead of blank UI
        with col2:
            st.caption("MST doesn’t use source/target.")
        with col3:
            st.caption("It returns a spanning tree across all nodes.")


    # Guardrail: Dijkstra requires non-negative weights
    # if (edges_clean["weight"] < 0).any():
    #     st.error("Negative edge weights detected → Dijkstra is not valid on this graph/weight.")
    #     st.stop()

    # if st.button("Run algorithm"):
    #     if algo.startswith("Dijkstra"):
    #         dist, prev, steps_df = dijkstra_steps(adj, nodes_list, source)

    #         st.subheader("Result")
    #         if target != "(none)":
    #             path = reconstruct_path(prev, source, target)
    #             if not path:
    #                 st.warning(f"No path found from {source} to {target}.")
    #             else:
    #                 st.write(f"**Shortest path:** {' → '.join(path)}")
    #                 st.write(f"**Total weight:** {dist[target]}")
    #         else:
    #             st.write("Computed shortest distances from source to all nodes.")

    #         st.subheader("Step-by-step table (iteration snapshots)")
    #         st.dataframe(steps_df, use_container_width=True)

    #         st.download_button(
    #             "Download steps (CSV)",
    #             data=steps_df.to_csv(index=False).encode("utf-8"),
    #             file_name="dijkstra_steps.csv",
    #             mime="text/csv"
    #         )

    #         # Also export final dist/prev
    #         out_df = pd.DataFrame({
    #             "node": nodes_list,
    #             "dist": [dist[n] for n in nodes_list],
    #             "prev": [prev[n] if prev[n] is not None else "" for n in nodes_list]
    #         })
    #         st.download_button(
    #             "Download final dist/prev (CSV)",
    #             data=out_df.to_csv(index=False).encode("utf-8"),
    #             file_name="dijkstra_result.csv",
    #             mime="text/csv"
    #         )
    
    has_negative = (edges_clean["weight"] < 0).any()
    if has_negative:
        st.warning("Negative weights detected: use Bellman–Ford / Floyd–Warshall. Dijkstra will be blocked.")
    
    run = st.button("Run algorithm")

    if run:
        directed = st.session_state.get("directed", True)
        nodes_list = nodes_clean["node_id"].astype(str).tolist()

        # Build representations
        adj = build_adj_list(edges_clean, directed=directed)
        edge_list = build_edge_list(edges_clean, directed=directed)

        # ---------- DIJKSTRA ----------
        if algo.startswith("Dijkstra"):
            if (edges_clean["weight"] < 0).any():
                st.error("Negative edge weights detected → Dijkstra is not valid on this weight.")
                st.stop()

            dist, prev, steps_df = dijkstra_steps(adj, nodes_list, source)

            st.subheader("Result")
            if target != "(none)":
                path = reconstruct_path(prev, source, target)
                if not path:
                    st.warning(f"No path found from {source} to {target}.")
                else:
                    st.write(f"**Shortest path:** {' → '.join(path)}")
                    st.write(f"**Total weight:** {dist[target]}")
            else:
                st.write("Computed shortest distances from source to all nodes.")

            st.subheader("Step-by-step table")
            st.dataframe(steps_df, use_container_width=True)

            st.download_button("Download steps (CSV)", steps_df.to_csv(index=False).encode("utf-8"),
                            "dijkstra_steps.csv", "text/csv")

        # ---------- BELLMAN-FORD ----------
        elif algo.startswith("Bellman"):
            dist, prev, steps_df, has_neg_cycle, neg_edges = bellman_ford_steps(nodes_list, edge_list, source)

            st.subheader("Result")
            if has_neg_cycle:
                st.error("Negative cycle detected (shortest paths undefined).")
                st.write("Edges that still relax on the extra check:")
                st.dataframe(pd.DataFrame(neg_edges, columns=["u","v","w"]), use_container_width=True)
            else:
                if target != "(none)":
                    path = reconstruct_path(prev, source, target)
                    if not path:
                        st.warning(f"No path found from {source} to {target}.")
                    else:
                        st.write(f"**Shortest path:** {' → '.join(path)}")
                        st.write(f"**Total weight:** {dist[target]}")
                else:
                    st.write("Computed shortest distances from source to all nodes.")

            st.subheader("Step-by-step table (iterations of relax-all-edges)")
            st.dataframe(steps_df, use_container_width=True)
            st.download_button("Download steps (CSV)", steps_df.to_csv(index=False).encode("utf-8"),
                            "bellman_ford_steps.csv", "text/csv")

        # ---------- FLOYD-WARSHALL ----------
        elif algo.startswith("Floyd"):
            keep_steps = st.checkbox("Keep k-step snapshots (can be heavy for big graphs)", value=False)
            dist_df, next_df, steps = floyd_warshall(nodes_list, edge_list, keep_steps=keep_steps)

            # Negative cycle check: dist[i][i] < 0 indicates a negative cycle
            if (np.diag(dist_df.values) < 0).any():
                st.error("Negative cycle detected (shortest paths undefined).")
            else:
                st.subheader("All-pairs shortest path distances")
                st.dataframe(dist_df, use_container_width=True)

                st.download_button("Download dist matrix (CSV)",
                                dist_df.to_csv(index=True).encode("utf-8"),
                                "floyd_warshall_dist.csv", "text/csv")

                st.download_button("Download next matrix (CSV)",
                                next_df.to_csv(index=True).encode("utf-8"),
                                "floyd_warshall_next.csv", "text/csv")

                if target != "(none)":
                    path = fw_reconstruct_path(next_df, source, target)
                    st.subheader("Path reconstruction (using next-matrix)")
                    if not path:
                        st.warning(f"No path found from {source} to {target}.")
                    else:
                        st.write(f"**Shortest path:** {' → '.join(path)}")
                        st.write(f"**Total weight:** {dist_df.loc[source, target]}")

                if keep_steps and steps:
                    st.subheader("k-step snapshots")
                    k = st.slider("View snapshot k", 0, len(steps)-1, 0)
                    st.dataframe(steps[k], use_container_width=True)

        # ---------- MST (KRUSKAL) ----------
        elif algo.startswith("MST"):
            if directed:
                st.error("MST requires an UNDIRECTED graph. Turn off 'Directed graph?' in the sidebar.")
                st.stop()

            mst_edges_df, total_w, steps_df = kruskal_mst_steps(nodes_list, edges_clean)
            
            st.caption(
                "MST (Kruskal): builds cheapest set of lanes connecting all facilities."
                if algo.startswith("MST") else
                "Shortest-path algorithms: compute best route(s) under the chosen weights."
            )


            st.subheader("Result")
            st.write(f"**Total MST weight:** {total_w}")
            st.write("**Edges in MST:**")
            st.dataframe(mst_edges_df, use_container_width=True)

            st.subheader("Step-by-step table (Kruskal selection)")
            st.dataframe(steps_df, use_container_width=True)

            st.download_button("Download MST edges (CSV)",
                            mst_edges_df.to_csv(index=False).encode("utf-8"),
                            "mst_edges.csv", "text/csv")

            st.download_button("Download MST steps (CSV)",
                            steps_df.to_csv(index=False).encode("utf-8"),
                            "mst_steps.csv", "text/csv")