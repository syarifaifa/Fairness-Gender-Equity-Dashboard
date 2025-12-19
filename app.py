import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Rasch ML & AI Dashboard", layout="wide")
st.title("Rasch Gender Equity Dashboard — ML & AI Bootcamp Edition")
st.write("""
This dashboard uses simulated 212-student responses to visualize Rasch-based assessment results.
""")

# -----------------------------
# CREATE STUDENT DATA WITH PACKAGE A & B
# -----------------------------

# Assume: 2 anchor items + 8 unique items per package
# Total 18 items: Item1-Item10 (Package A, anchor: Item9-10)
# Total 18 items: Item11-Item18 (Package B, anchor: Item9-10)

students = []
for i in range(1, 213):
    gender = "Male" if i <= 106 else "Female"
    package = "A" if i % 2 == 1 else "B"
    answers = []
    for item in range(1, 19):
        # Anchor items: Item9 & 10, both packages answer them
        if item in [9,10]:
            answers.append(1)  # keep correct like original
        else:
            if (package == "A" and item <= 8) or (package == "B" and item >= 11):
                answers.append(1)  # correct for simplicity
            else:
                answers.append(np.nan)  # not answered because other package
    students.append([f"{i}-{'L' if gender=='Male' else 'P'}"] + answers + [gender])

df = pd.DataFrame(students, columns=["ID"] + [f"Item{i}" for i in range(1,19)] + ["Gender"])

# Raw score
df["RawScore"] = df.iloc[:,1:19].sum(axis=1)

# Estimate ability (simple linear transform)
df["Ability"] = (df["RawScore"] - df["RawScore"].mean()) / df["RawScore"].std()

# -----------------------------
# Student Count
# -----------------------------
st.header("1. Student Gender Distribution")
df_counts = df["Gender"].value_counts().reset_index()
df_counts.columns = ["Gender", "Count"]

chart_gender = alt.Chart(df_counts).mark_bar(color="#4C72B0").encode(
    x="Gender:N",
    y="Count:Q",
    tooltip=["Gender","Count"]
)
st.altair_chart(chart_gender, use_container_width=True)
st.caption(f"Total students: {df.shape[0]} (Male {df_counts.loc[df_counts['Gender']=='Male','Count'].values[0]}, Female {df_counts.loc[df_counts['Gender']=='Female','Count'].values[0]})")

# -----------------------------
# Item Difficulty (simulated same as original)
# -----------------------------
st.header("2. Item Difficulty")
item_difficulty = np.array([-0.18, -0.12, -0.09, -0.06, -0.03, 0.00,
                            0.02, 0.04, 0.05, 0.07, 0.09, 0.10,
                            0.12, 0.14, 0.16, 0.18, 0.20, 0.23])
df_diff = pd.DataFrame({"Item":[f"Item{i}" for i in range(1,19)],
                        "Difficulty":item_difficulty})
diff_chart = alt.Chart(df_diff).mark_bar(color="#4C72B0").encode(
    x="Item:N",
    y="Difficulty:Q",
    tooltip=["Item","Difficulty"]
)
st.altair_chart(diff_chart, use_container_width=True)
st.caption("Item difficulty range: −0.18 to +0.23 logits, mean ≈ 0.00")

# -----------------------------
# DIF Analysis (simulated same as original)
# -----------------------------
st.header("3. Differential Item Functioning (Gender)")
dif_values = np.array([-0.08, -0.07, -0.06, -0.05, -0.04, -0.03,
                       -0.02, -0.01, 0.00, 0.01, 0.02, 0.03,
                        0.04, 0.05, 0.06, 0.07, 0.08, -0.02])
df_dif = pd.DataFrame({"Item":[f"Item{i}" for i in range(1,19)],
                       "DIF":dif_values})
base = alt.Chart(df_dif).mark_bar().encode(
    x="Item:N",
    y=alt.Y("DIF:Q", title="DIF Contrast (logits)"),
    color=alt.condition(alt.datum.DIF>=0, alt.value("#1B9E77"), alt.value("#D95F02")),
    tooltip=["Item","DIF"]
)
threshold = alt.Chart(pd.DataFrame({"y":[0.5,-0.5]})).mark_rule(strokeDash=[5,5], color="red").encode(y="y")
st.altair_chart(base + threshold, use_container_width=True)
st.caption("DIF Contrast Range (Logits): −0.08 to +0.08 — No significant gender bias detected.")

# -----------------------------
# Summary Metrics
# -----------------------------
st.header("4. Summary Metrics (Rasch)")
st.write("""
**Item Statistics**
- Item Reliability: 0.95
- Item Separation: 4.48
- RMSE: 0.04
- Infit & Outfit MNSQ: 0.5 – 1.5

**Person Statistics**
- Person Reliability: 0.91
- Person Separation: 3.10
- Cronbach Alpha: 0.78

**Gender Fairness**
- DIF Contrast Range: −0.08 to +0.08
- Significant DIF Items: 0
""")

# -----------------------------
# RESUME / CONCLUSION
# -----------------------------
st.header("7. Resume & Conclusion")

st.success("""
**Key Findings from Rasch Analysis (212 Students):**

1. **Balanced Gender Participation**  
   The assessment involved **212 students**, consisting of **106 male** and **106 female** students, 
   ensuring equal representation across genders.

2. **Well-Targeted Item Difficulty**  
   Item difficulty ranged from **−0.18 to +0.23 logits**, with a mean centered at **0.00 logits**, 
   indicating that the test was optimally aligned with the average student ability.

3. **High Measurement Quality**  
   - Item Reliability: **0.95**  
   - Item Separation: **4.48**  
   - Person Reliability: **0.91**  
   These values indicate strong measurement precision and clear differentiation among student abilities.

4. **Good Model–Data Fit**  
   Infit and Outfit MNSQ statistics fell within the acceptable range (**0.5–1.5**), 
   supporting the validity of the Rasch measurement model.

5. **Gender Fairness Confirmed**  
   Differential Item Functioning (DIF) analysis showed a contrast range of **−0.08 to +0.08 logits**, 
   with **no statistically significant gender bias** detected across all 18 items.

**Conclusion:**  
This Rasch-based assessment demonstrates **high reliability, good construct validity, 
and gender-neutral measurement**, making it suitable for fair and equitable evaluation 
in educational contexts.
""")

