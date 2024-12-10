# %%
import json
import pandas as pd
# %%
df = pd.read_csv('metadata.csv')
data = dict(title='Synthetic CT Datasets of Intracranial Hemorrhage',
            columns=df.columns.to_list(),
            references=[
                dict(title="[code]", url='https://github.com/DIDSR/PedSilicoICH')
                ],
            images=[])

for idx, row in df.iterrows():
    data['images'].append(row.to_dict())

with open("metadata.json", "w") as outfile:
    json.dump(data, outfile)
# %%
