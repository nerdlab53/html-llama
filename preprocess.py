import pandas as pd

df = pd.read_csv('data.csv')
df = df.fillna("")

def generate_text(text):
    return f"""<s> [INST]{text['label']} [/INST] {text['html']} </s>"""

final = []
for _, row in df.iterrows():
    pre_final = generate_text(row)
    final.append(pre_final)

df.loc[:, 'text'] = final
new_data = df['text'].to_csv("preprocessed_data.csv", index=False)