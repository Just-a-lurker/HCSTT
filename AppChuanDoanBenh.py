import streamlit as st
import pandas as pd
from typing import List, Set, Tuple
from collections import deque
import networkx as nx

INF = float("inf")
import re

# === Đọc bảng thay thế Anh -> Việt ===
replace_df = pd.read_csv("translate.csv", encoding="utf-8")
replace_map = dict(zip(replace_df["en"].str.strip().str.lower(),
                       replace_df["vi"].str.strip()))

def vi_en(word: str) -> str:
    """Trả về 'vi - en' nếu có trong bảng, ngược lại trả lại nguyên từ"""
    if not isinstance(word, str):
        return word
    w_lower = word.strip().lower()
    if w_lower in replace_map:
        return f"{replace_map[w_lower]} - {word.strip()}"
    return word.strip()


# ================== CÁC HÀM CƠ SỞ (đã có từ hệ suy diễn) ==================
def build_adj(R_all: List[Tuple[Set[str], Set[str]]]) -> dict:
    G = {}
    for left, right in R_all:
        for l in left:
            G.setdefault(l, set()).update(right)
        for r in right:
            G.setdefault(r, set())
    return G

def dist_to_set(G: dict, src: str, targets: Set[str]) -> float:
    if src in targets:
        return 0
    q = deque([(src, 0)])
    vis = {src}
    while q:
        node, d = q.popleft()
        for nxt in G.get(node, ()):
            if nxt in targets:
                return d + 1
            if nxt not in vis:
                vis.add(nxt)
                q.append((nxt, d + 1))
    return float("inf")

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

def rule_to_str(rule):
    l, r = rule
    return f"{' ^ '.join(sorted(l))} -> {' ^ '.join(sorted(r))}"

def rule_name(idx, rule):
    l, r = rule
    return f"r{idx}: {' ^ '.join(sorted(l))} -> {' ^ '.join(sorted(r))}"

# ================== SUY DIỄN TIẾN FPG ==================
def suy_dien_tien_FPG(TG: Set[str], R_all: List[Tuple[Set[str], Set[str]]], KL: Set[str]):
    TG0 = set(TG)
    R_remain = list(R_all)[:]
    G = build_adj(R_all)
    VET = []
    steps = []

    THOA = [r for r in R_remain if r[0].issubset(TG) and not r[1].issubset(TG)]

    steps.append({
        "Bước": 0,
        "THOA": [rule_to_str(r) for r in THOA],
        "TG": ", ".join(sorted(TG)),
        "R": [rule_to_str(r) for r in R_remain],
        "VET": "",
        "Candidates": ""
    })

    step_id = 0
    while THOA and not KL.issubset(TG):
        step_id += 1
        cand_info = []
        scored = []
        for r in THOA:
            left, right = r
            h_vals = []
            for qnode in right:
                d = dist_to_set(G, qnode, KL)
                h_vals.append(d if d != float("inf") else INF)
            h_r = min(h_vals) if h_vals else INF
            idx = R_remain.index(r)
            name = rule_name(idx + 1, r)
            cand_info.append(f"{name}={h_r if h_r < INF else '∞'}")
            scored.append((h_r, idx, r))

        scored.sort(key=lambda x: (x[0], x[1]))
        chosen_h, chosen_idx, chosen_rule = scored[0]
        chosen_name = rule_name(chosen_idx + 1, chosen_rule)
        cand_str = ", ".join(cand_info)
        cand_str += f"    => Chọn: {chosen_name} (h={chosen_h if chosen_h < INF else '∞'})"

        steps.append({
            "Bước": step_id,
            "THOA": [rule_to_str(r) for r in THOA],
            "TG": ", ".join(sorted(TG)),
            "R": [rule_to_str(r) for r in R_remain],
            "VET": ", ".join([rule_to_str(x) for x in VET]),
            "Candidates": cand_str
        })

        if chosen_h >= INF:
            break

        left, right = chosen_rule
        VET.append(chosen_rule)
        TG |= right
        R_remain.remove(chosen_rule)
        THOA = [r for r in R_remain if r[0].issubset(TG) and not r[1].issubset(TG)]

    TG_kq = TG | set().union(*[r[1] for r in VET])
    return KL.issubset(TG_kq), TG_kq, VET, steps


# ================== HÀM TẠO LUẬT TỪ CSV ==================
def load_rules_from_csv(dataset):
    """Tạo luật (Symptom -> Disease) từ file dataset chứa tên triệu chứng trực tiếp"""
    R_all = []
    symptom_cols = [c for c in dataset.columns if c.startswith("Symptom")]
    for _, row in dataset.iterrows():
        disease = row["Disease"].strip()
        left = set()
        for col in symptom_cols:
            val = str(row[col]).strip()
            if val and val.lower() != "nan" and val != "0":
                left.add(val)
        if left:
            R_all.append((left, {disease}))
    return R_all



# ================== APP STREAMLIT ==================
st.set_page_config("Hệ chẩn đoán bệnh bằng Suy diễn", layout="wide")

st.title("HỆ CHUẨN ĐOÁN BỆNH DỰA TRÊN SUY DIỄN (FPG)")

dataset = pd.read_csv("dataset.csv",  encoding='utf-8')

symptom_severity = pd.read_csv("Symptom-severity.csv", encoding='utf-8')
precaution = pd.read_csv("symptom_precaution.csv",encoding='utf-8')
description = pd.read_csv("symptom_Description.csv", encoding='utf-8', engine='python')
description["Description"] = description["Description"].astype(str).replace({'\r\n': '\n', '\\n': '\n'}, regex=True)
for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
    precaution[col] = precaution[col].astype(str).replace({'\r\n': '\n', '\\n': '\n'}, regex=True)


# Map số -> triệu chứng
sym_map = {i + 1: symptom_severity.iloc[i]["Symptom"] for i in range(len(symptom_severity))}

# Tạo luật
R_all = load_rules_from_csv(dataset)

# ============= GIAO DIỆN NGƯỜI DÙNG =============
diseases = sorted(dataset["Disease"].unique())

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Nhập tên bệnh để gợi ý")
    query = st.text_input("Tìm kiếm bệnh", "")
    if query:
        gợi_ý = [d for d in diseases if query.lower() in d.lower()]
        st.write("**Gợi ý:**", gợi_ý if gợi_ý else "Không tìm thấy")

    st.subheader("Chọn triệu chứng bạn đang có:")
    symptoms_display = [vi_en(s) for s in sorted(symptom_severity["Symptom"].tolist())]
    selected_display = st.multiselect("Triệu chứng:", symptoms_display)

    # Chuyển lại sang tiếng Anh để xử lý nội bộ
    selected_symptoms = [s.split(" - ")[-1] if " - " in s else s for s in selected_display]

with col2:
    if st.button("Chuẩn đoán"):
        if not selected_symptoms:
            st.warning("Chưa chọn triệu chứng")
        else:
            st.subheader("KẾT QUẢ DỰ ĐOÁN DỰA TRÊN TRIỆU CHỨNG")

            # Map triệu chứng -> trọng số
            weight_map = dict(zip(symptom_severity["Symptom"].str.strip(), symptom_severity["weight"]))

            # Gom tất cả triệu chứng của mỗi bệnh (vì mỗi bệnh chỉ 1 dòng)
            disease_symptoms = {}
            symptom_columns = [c for c in dataset.columns if c.startswith("Symptom")]

            for _, row in dataset.iterrows():
                disease = row["Disease"].strip()
                symptoms = set()
                for col in symptom_columns:
                    val = str(row[col]).strip()
                    if val and val.lower() != "nan" and val != "0":
                        symptoms.add(val)
                if symptoms:
                    disease_symptoms[disease] = symptoms

            # Tính điểm khớp có trọng số
            scores = {}
            for disease, sym_set in disease_symptoms.items():
                matched = set(selected_symptoms) & sym_set
                if matched:
                    total_weight = sum(weight_map.get(s, 0) for s in sym_set)
                    matched_weight = sum(weight_map.get(s, 0) for s in matched)
                    match_score = matched_weight / total_weight if total_weight > 0 else 0
                    scores[disease] = match_score

            # Kết quả
            if not scores:
                st.error("Không tìm thấy bệnh phù hợp với triệu chứng đã chọn.")
            else:
                sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_disease, top_score = sorted_diseases[0]

                st.success(
                    f"Bệnh có khả năng cao nhất: {vi_en(top_disease)} (Độ khớp có trọng số: {top_score * 100:.1f}%)")

                desc = description.loc[description["Disease"] == top_disease, "Description"].values
                pre = precaution.loc[precaution["Disease"] == top_disease, ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].values
                if len(desc) > 0:
                    st.markdown(f"**Mô tả:**<br>{desc[0].replace(chr(10), '<br>')}", unsafe_allow_html=True)
                if len(pre) > 0:
                    pre_clean = [p.replace('\n', '<br>') for p in pre[0] if isinstance(p, str)]
                    st.markdown(f"**Khuyến nghị:**<br>{'<br>'.join(pre_clean)}", unsafe_allow_html=True)

                # Hiển thị các bệnh khác
                if len(sorted_diseases) > 1:
                    st.write("### Các bệnh có thể khác:")
                    for disease, score in sorted_diseases[1:]:
                        st.write(f"- {vi_en(disease)} (Độ khớp có trọng số: {score * 100:.1f}%)")
