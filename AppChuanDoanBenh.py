import streamlit as st
import pandas as pd
from typing import List, Set, Tuple

# ================== Hàm hỗ trợ ==================
def vi_en(word: str) -> str:
    """Trả về 'vi - en' nếu có trong bảng, ngược lại trả lại nguyên từ"""
    if not isinstance(word, str):
        return word
    w_lower = word.strip().lower()
    if w_lower in replace_map:
        return f"{replace_map[w_lower]} - {word.strip()}"
    return word.strip()

def clean_symptom(s: str) -> str:
    return s.strip().replace(" ", "_").lower()

# ================== Forward Chaining ==================
def forward_chaining(TG: Set[str], R_all: List[Tuple[Set[str], str]], weight_map: dict):
    TG = set(clean_symptom(s) for s in TG)
    disease_scores = {}
    VET = []

    for idx, (left, disease) in enumerate(R_all, 1):
        matched = TG & left
        if matched:
            total_weight = sum(weight_map.get(s, 1) for s in left)
            matched_weight = sum(weight_map.get(s, 1) for s in matched)
            score = matched_weight / total_weight
            if disease in disease_scores:
                disease_scores[disease].append(score)
            else:
                disease_scores[disease] = [score]
            VET.append((left, disease, score))

    # Tính trung bình score mỗi bệnh
    avg_scores = {d: sum(scores)/len(scores) for d, scores in disease_scores.items()}
    sorted_diseases = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_diseases, VET

# ================== App Streamlit ==================
st.set_page_config("Hệ chẩn đoán bệnh bằng Suy diễn tiến", layout="wide")
st.title("HỆ CHUẨN ĐOÁN BỆNH DỰA TRÊN SUY DIỄN TIẾN")

# --- Load dữ liệu ---
dataset = pd.read_csv("dataset.csv", encoding='utf-8')
symptom_severity = pd.read_csv("Symptom-severity.csv", encoding='utf-8')
precaution = pd.read_csv("symptom_precaution.csv", encoding='utf-8')
description = pd.read_csv("symptom_Description.csv", encoding='utf-8', engine='python')
description["Description"] = description["Description"].astype(str).replace({'\r\n':'\n','\\n':'\n'}, regex=True)
for col in ["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]:
    precaution[col] = precaution[col].astype(str).replace({'\r\n':'\n','\\n':'\n'}, regex=True)

# Bảng Anh -> Việt
replace_df = pd.read_csv("translate.csv", encoding="utf-8")
replace_map = dict(zip(replace_df["en"].str.strip().str.lower(), replace_df["vi"].str.strip()))

# Map trọng số
weight_map = {clean_symptom(s): w for s, w in zip(symptom_severity["Symptom"], symptom_severity["weight"])}

# Tạo luật: mỗi dòng là 1 luật
def load_rules_per_row(dataset):
    R_all = []
    symptom_cols = [c for c in dataset.columns if c.startswith("Symptom")]
    for _, row in dataset.iterrows():
        disease = row["Disease"].strip()
        left = set()
        for col in symptom_cols:
            val = str(row[col]).strip()
            if val and val.lower() != "nan" and val != "0":
                left.add(clean_symptom(val))
        if left:
            R_all.append((left, disease))
    return R_all

R_all = load_rules_per_row(dataset)

# ================== Giao diện ==================
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Chọn triệu chứng bạn đang có:")
    symptoms_display = [vi_en(s) for s in sorted(symptom_severity["Symptom"].tolist())]
    selected_display = st.multiselect("Triệu chứng:", symptoms_display)
    selected_symptoms = [s.split(" - ")[-1] if " - " in s else s for s in selected_display]
    # if selected_display:
    #     st.markdown("**Triệu chứng đã chọn:**")
    #     for s in selected_display:
    #         st.markdown(f"- {s}")
    # else:
    #     st.markdown("_Chưa chọn triệu chứng nào._")

with col2:
    if st.button("Chuẩn đoán"):
        if not selected_symptoms:
            st.warning("Chưa chọn triệu chứng")
        else:
            sorted_diseases, VET = forward_chaining(selected_symptoms, R_all, weight_map)

            if not sorted_diseases:
                st.error("Không tìm thấy bệnh phù hợp với triệu chứng đã chọn.")
            else:
                # --- Bệnh cao nhất ---
                top_disease, top_score = sorted_diseases[0]
                st.success(f"Bệnh có khả năng cao nhất: {vi_en(top_disease)} (Độ khớp có trọng số: {top_score*100:.1f}%)")

                # Triệu chứng gây bệnh
                related_symptoms = set()
                for left, d, score in VET:
                    if d == top_disease:
                        related_symptoms |= left
                st.write("**Triệu chứng liên quan:**", ", ".join([vi_en(s) for s in sorted(related_symptoms)]))

                # Description + Precaution
                desc = description.loc[description["Disease"]==top_disease,"Description"].values
                pre = precaution.loc[precaution["Disease"]==top_disease, ["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]].values
                if len(desc)>0:
                    st.markdown(f"**Mô tả:**<br>{desc[0].replace(chr(10), '<br>')}", unsafe_allow_html=True)
                if len(pre)>0:
                    pre_clean = [p.replace('\n','<br>') for p in pre[0] if isinstance(p,str)]
                    st.markdown(f"**Khuyến nghị:**<br>{'<br>'.join(pre_clean)}", unsafe_allow_html=True)

                # --- Các bệnh khả năng khác ---
                if len(sorted_diseases) > 1:
                    st.write("### Các bệnh có khả năng khác:")
                    for disease, score in sorted_diseases[1:]:
                        with st.expander(f"{vi_en(disease)} (Độ khớp: {score*100:.1f}%)"):
                            # Triệu chứng liên quan
                            related_symptoms = set()
                            for left, d, s in VET:
                                if d == disease:
                                    related_symptoms |= left
                            st.write("**Triệu chứng liên quan:**", ", ".join([vi_en(s) for s in sorted(related_symptoms)]))
                            # Description + Precaution
                            desc = description.loc[description["Disease"]==disease,"Description"].values
                            pre = precaution.loc[precaution["Disease"]==disease, ["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]].values
                            if len(desc)>0:
                                st.markdown(f"**Mô tả:**<br>{desc[0].replace(chr(10), '<br>')}", unsafe_allow_html=True)
                            if len(pre)>0:
                                pre_clean = [p.replace('\n','<br>') for p in pre[0] if isinstance(p,str)]
                                st.markdown(f"**Khuyến nghị:**<br>{'<br>'.join(pre_clean)}", unsafe_allow_html=True)
