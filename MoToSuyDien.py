# suy_dien_tamgiac.py
from typing import List, Tuple, Set
import json
from pyvis.network import Network
from collections import deque
from collections import deque
from typing import List, Tuple, Set

Rule = Tuple[Set[str], Set[str]]

with open("congthuc.json", "r", encoding="utf-8") as f:
    formulas = json.load(f)

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
            if not line:
                continue
            if line.upper().startswith("R:"):
                continue
            if "->" in line:
                R.append(parse_rule(line))
    return R

def loc(TG: Set[str], R: List[Rule]) -> List[Rule]:
    return [r for r in R if r[0].issubset(TG) and not r[1].issubset(TG)]

def suy_dien_tien(TG: Set[str], R: List[Rule], KL: Set[str]):
    TG0 = set(TG)
    VET: List[Rule] = []
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
    used: List[Rule] = []
    for left, right in reversed(VET):
        if right & needed:
            used.insert(0, (left, right))
            needed |= left
    return used

def reconstruct_path(TG0: Set[str], used_rules: List[Rule]) -> List[str]:
    TG_cur = set(TG0)
    path = list(TG0)
    for left, right in used_rules:
        new = sorted(list(right - TG_cur))
        if new:
            path.extend(new)
        TG_cur |= right
    return path

# ----------- VẼ FPG -----------
def ve_FPG_interactive_edges(TG0, KL, R_all, output_html="fpg.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.set_options("""{ "physics": { "enabled": false } }""")

    # gom toàn bộ sự kiện từ luật gốc
    events = set(TG0) | KL
    for left, right in R_all:
        events |= left
        events |= right

    for e in events:
        if e in TG0:
            net.add_node(e, label=e, shape="ellipse", color="lightblue")
        elif e in KL:
            net.add_node(e, label=e, shape="ellipse", color="lightgray")
        else:
            net.add_node(e, label=e, shape="ellipse", color="lightgreen")

    for idx, (left, right) in enumerate(R_all, start=1):
        rule_str = f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}"
        label = f"r{idx}"
        for l in left:
            for r in right:
                net.add_edge(l, r, arrows="to", label=label)

    net.write_html(output_html)
    print(f"Đã tạo file {output_html} (FPG).")

# ----------- VẼ RPG -----------
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
    print(f"Đã tạo file {output_html} (RPG).")

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



# ----------- MAIN -----------
if __name__ == "__main__":
    mode = input("Chọn mode (1 = TG,KL,R từ file | 2 = R từ file, TG,KL nhập tay): ").strip()
    if mode == "1":
        file_path = input("File (TG:, KL:, R:): ").strip() or "dulieu.txt"
        TG, KL, R = set(), set(), []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        mode2 = None
        for line in lines:
            if line.startswith("TG:"):
                TG = set(line.replace("TG:", "").strip().split())
            elif line.startswith("KL:"):
                KL = set(line.replace("KL:", "").strip().split())
            elif line.startswith("R:"):
                mode2 = "R"
            elif mode2 == "R":
                R.append(parse_rule(line))
    elif mode == "2":
        file_path = input("File luật (chứa R: và các luật): ").strip() or "luat.txt"
        R = doc_luat(file_path)
        TG = set(input("Nhập TG: ").split())
        KL = set(input("Nhập KL: ").split())
    else:
        print("Mode không hợp lệ")
        exit(1)

    # suy diễn tiến
    ok, TG_kq, VET, TG0 = suy_dien_tien(set(TG), list(R), set(KL))
    print("=== Suy diễn tiến ===")
    print("Thành công:", ok)
    print("Tập sự kiện cuối cùng:", TG_kq)
    used_rules = prune_vet(VET, TG0, KL)
    path_min = reconstruct_path(TG0, used_rules)
    for left, right in used_rules:
        rule_str = f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}"
        formula = formulas.get(rule_str, "")
        if formula:
            print(f"{rule_str}    ({formula})")
        else:
            print(rule_str)
    print("Đường suy diễn tiến:", " -> ".join(path_min))

    # suy diễn lùi
    ok_b, goals_b, VEL = suy_dien_lui(set(TG), list(R), set(KL))
    print("\n=== Suy diễn lùi ===")
    print("Thành công:", ok_b)
    print("Mục tiêu còn lại:", goals_b)
    for left, right in VEL:
        rule_str = f"{' ^ '.join(sorted(left))} -> {' ^ '.join(sorted(right))}"
        print(rule_str)

    # xuất FPG và RPG
    ve_FPG_interactive_edges(TG, KL, R, "fpg.html")
    ve_RPG(R, "rpg.html")
