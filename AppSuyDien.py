import streamlit as st
from typing import List, Tuple, Set
from pyvis.network import Network
import tempfile
import os
import pandas as pd
from collections import deque
from typing import List, Tuple, Set

Rule = Tuple[Set[str], Set[str]]

# ------------------ PARSE & DOC LUẬT ------------------
def parse_rule(line: str) -> Rule:
    line = line.strip()
    line = line.replace("->", " -> ").replace("^", " ")
    left_s, right_s = line.split("->")
    left = set(left_s.strip().split())
    right = set(right_s.strip().split())
    return left, right

def doc_luat(file_path: str) -> List[Rule]:
    R = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "->" not in line:
                continue
            R.append(parse_rule(line))
    return R

def rule_to_str(r: Rule) -> str:
    left, right = r
    return f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}"

# ------------------ SUY DIỄN ------------------
def suy_dien_tien_bt(TG: Set[str], R: List[Rule], KL: Set[str], order="min", depth=0, max_depth=2000):
    if KL.issubset(TG):
        return True, TG, []

    if depth > max_depth:
        return False, TG, []

    indices = range(len(R)) if order == "min" else range(len(R) - 1, -1, -1)
    for i in indices:
        left, right = R[i]
        if left.issubset(TG) and not right.issubset(TG):
            new_TG = set(TG) | set(right)
            ok, final_TG, used = suy_dien_tien_bt(new_TG, R, KL, order, depth + 1, max_depth)
            if ok:
                return True, final_TG, [(left, right)] + used
    return False, TG, []

def suy_dien_lui_bt(TG: Set[str], R: List[Rule], KL: Set[str], order="min", depth=0, max_depth=2000):
    if KL.issubset(TG):
        return True, TG, []
    if depth > max_depth:
        return False, KL, []

    goals = set(KL)
    indices = range(len(R)) if order == "min" else range(len(R) - 1, -1, -1)
    for i in indices:
        left, right = R[i]
        if right & goals:
            new_goals = (goals - (right & goals)) | left
            ok, _, used = suy_dien_lui_bt(TG, R, new_goals, order, depth + 1, max_depth)
            if ok:
                return True, set(), [(left, right)] + used
    return False, goals, []


def loc(TG: Set[str], R: List[Rule]) -> List[Rule]:
    return [r for r in R if r[0].issubset(TG) and not r[1].issubset(TG)]

def suy_dien_tien(TG: Set[str], R: List[Rule], KL: Set[str]):
    TG0 = set(TG)
    VET = []
    THOA = loc(TG, R)
    while THOA and not KL.issubset(TG):
        r = THOA.pop(0)
        left, right = r
        VET.append(r)
        TG |= right
        R.remove(r)
        THOA = loc(TG, R)
    return KL.issubset(TG), TG, VET, TG0

def prune_vet(VET: List[Rule], TG0: Set[str], KL: Set[str]) -> List[Rule]:
    needed = set(KL)
    used = []
    for left, right in reversed(VET):
        if right & needed:
            used.insert(0, (left, right))
            needed |= left
    return used

# Hàm phụ
def build_adj(R_all: List[Tuple[Set[str], Set[str]]]) -> dict:
    G = {}
    for left, right in R_all:
        for l in left:
            G.setdefault(l, set()).update(right)
        for r in right:
            G.setdefault(r, set())
    return G

def shortest_distance(G: dict, GT: Set[str], target: str) -> float:
    if target in GT:
        return 0
    visited = set(GT)
    q = deque([(g, 0) for g in GT])
    while q:
        node, dist = q.popleft()
        for nxt in G.get(node, ()):
            if nxt == target:
                return dist + 1
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, dist + 1))
    return float("inf")

def h_rule(left: Set[str], G: dict, GT: Set[str]) -> float:
    if not left:
        return float("inf")
    return max(shortest_distance(G, GT, f) for f in left)

# Backtracking backward chaining
def suy_dien_lui(TG: Set[str], R_all: List[Tuple[Set[str], Set[str]]], KL: Set[str],
                 max_depth: int = 2000):
    """
    Backward chaining with backtracking + heuristic h(r,GT).
    Trả về (ok:bool, remaining_goals:set, selected_rules:list).
    """
    G = build_adj(R_all)           # đồ thị FPG cố định (dùng cho h)
    R_list = list(R_all)           # danh sách luật (không xoá gốc)
    visited_states = set()         # memo các trạng thái goals đã thăm
    best_solution = None
    best_remaining = None          # trạng thái còn lại nhỏ nhất tìm được

    path: List[Tuple[Set[str], Set[str]]] = []

    def dfs(goals: Set[str], depth: int) -> bool:
        nonlocal best_solution, best_remaining
        if depth > max_depth:
            return False
        state = frozenset(goals)
        if state in visited_states:
            return False
        visited_states.add(state)

        # nếu đã đạt
        if goals.issubset(TG):
            best_solution = path.copy()
            best_remaining = set()
            return True

        # cập nhật best_remaining (để báo lại khi thất bại hoàn toàn)
        unresolved = {g for g in goals if g not in TG}
        if best_remaining is None or len(unresolved) < len(best_remaining):
            best_remaining = set(unresolved)

        # chọn một goal chưa đạt để expand: chọn goal có ít ứng viên nhất (fail-first)
        candidates_per_goal = []
        for g in unresolved:
            cands = [r for r in R_list if g in r[1]]
            candidates_per_goal.append((g, cands))
        # nếu không có luật nào sinh goal thì dead-end
        if not candidates_per_goal:
            return False

        # chọn goal có ít ứng viên; tie-break bằng min h(candidate)
        candidates_per_goal.sort(key=lambda item: (len(item[1]),
            min((h_rule(r[0], G, TG) for r in item[1]), default=float("inf"))))
        chosen_goal, candidates = candidates_per_goal[0]

        # sắp các luật ứng viên theo h(r,GT) tăng dần, ưu luật ít tiền đề
        candidates_sorted = sorted(candidates, key=lambda r: (h_rule(r[0], G, TG), len(r[0])))

        for r in candidates_sorted:
            left, right = r
            # áp dụng: thay các mục tiêu hiện có thuộc right bằng left
            new_goals = (goals - (right & goals)) | left
            path.append(r)
            ok = dfs(new_goals, depth + 1)
            if ok:
                return True
            path.pop()
        return False

    start_goals = set(KL)
    success = dfs(start_goals, 0)

    if success and best_solution is not None:
        return True, set(), best_solution
    else:
        # trả về trạng thái còn lại tối thiểu đã tìm được và luật đã thử (có thể rỗng)
        return False, best_remaining or (set(KL) - TG), path


# ------------------ VẼ FPG ------------------
def ve_FPG_interactive_edges(TG0, KL, R_all, output_html="fpg.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.set_options("""{ "physics": { "enabled": false } }""")

    events = set(TG0) | KL
    for left, right in R_all:
        events |= left
        events |= right

    for e in events:
        color = "lightgreen"
        if e in TG0:
            color = "lightblue"
        elif e in KL:
            color = "lightgray"
        net.add_node(e, label=e, title=e, shape="circle", color=color)

    for idx, (left, right) in enumerate(R_all, start=1):
        #label = f"r{idx}"
        for l in left:
            for r in right:
                net.add_edge(l, r, arrows="to")
    net.write_html(output_html)
    return output_html

# ------------------ VẼ RPG ------------------
def ve_RPG(R_all, output_html="rpg.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.set_options("""{ "physics": { "enabled": false } }""")

    rule_nodes = []
    for idx, (left, right) in enumerate(R_all, start=1):
        rname = f"r{idx}"
        net.add_node(rname, shape="box", color="orange")
        rule_nodes.append((rname, left, right))

    for i, (ri, left_i, right_i) in enumerate(rule_nodes):
        for j, (rj, left_j, right_j) in enumerate(rule_nodes):
            if i == j:
                continue
            if right_i & left_j:
                net.add_edge(ri, rj, arrows="to")
    net.write_html(output_html)
    return output_html

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="Hệ suy diễn", layout="wide")
st.title("HỆ SUY DIỄN")

uploaded = st.file_uploader("Tải lên file luật (.txt)", type=["txt"])

if "rules_df" not in st.session_state:
    st.session_state["rules_df"] = pd.DataFrame(columns=["Vế trái", "Vế phải"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    R = doc_luat(tmp_path)
    st.session_state["rules_df"] = pd.DataFrame(
        [{"Vế trái": " ^ ".join(sorted(l)), "Vế phải": " ^ ".join(sorted(r))} for l, r in R]
    )

# --- Hiển thị bảng luật ---
st.subheader("Bảng luật (có thể chỉnh sửa)")
rules_df = st.data_editor(
    st.session_state["rules_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="editable_rules"
)

# --- Nút cập nhật bảng luật ---
if st.button("Cập nhật bảng luật"):
    st.session_state["rules_df"] = rules_df
    st.success("Đã cập nhật bảng luật vào bộ nhớ.")


# Tạo danh sách R mới
R = []
for _, row in rules_df.iterrows():
    left_str = str(row.get("Vế trái") or "").strip()
    right_str = str(row.get("Vế phải") or "").strip()
    if not left_str or not right_str:
        continue
    left = set(left_str.replace("^", " ").split())
    right = set(right_str.replace("^", " ").split())
    if left and right:
        R.append((left, right))


TG_input = st.text_input("Nhập giả thuyết ban đầu (GT):", "")
KL_input = st.text_input("Nhập kết luận cần chứng minh (KL):", "")

if TG_input and KL_input:
    TG = set(TG_input.split())
    KL = set(KL_input.split())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Suy diễn tiến")
        order_tien = st.radio("Thứ tự duyệt luật", ["min", "max"], key="order_tien", horizontal=True)
        if st.button("Chạy suy diễn tiến"):
            ok, TG_kq, VET = suy_dien_tien_bt(set(TG), list(R), set(KL), order=order_tien)
            st.write("**Thành công:**", ok)
            st.write("**Tập sự kiện cuối cùng:**", TG_kq)
            st.write("**Các luật đã dùng:**")
            for left, right in VET:
                st.write(f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}")

    with c2:
        st.markdown("### Suy diễn lùi")
        order_lui = st.radio("Thứ tự duyệt luật", ["min", "max"], key="order_lui", horizontal=True)
        if st.button("Chạy suy diễn lùi"):
            ok, goals, VEL = suy_dien_lui_bt(set(TG), list(R), set(KL), order=order_lui)
            st.write("**Thành công:**", ok)
            st.write("**Mục tiêu còn lại:**", goals)
            st.write("**Các luật đã dùng:**")
            for left, right in VEL:
                st.write(f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}")

    with c3:
        if st.button("Lưu file luật đã chỉnh sửa"):
            edited_path = "luat_chinh_sua.txt"
            with open(edited_path, "w", encoding="utf-8") as f:
                for left, right in R:
                    f.write(f"{' ^ '.join(sorted(left))}->{ ' ^ '.join(sorted(right))}\n")
            st.success(f"Đã lưu file: {edited_path}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Vẽ sơ đồ FPG"):
            fpg_html = ve_FPG_interactive_edges(TG, KL, R)
            st.components.v1.html(open(fpg_html, encoding="utf-8").read(), height=750, scrolling=True)
    with col2:
        if st.button("Vẽ sơ đồ RPG"):
            rpg_html = ve_RPG(R)
            st.components.v1.html(open(rpg_html, encoding="utf-8").read(), height=750, scrolling=True)
