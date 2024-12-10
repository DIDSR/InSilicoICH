# %%
import json
import pandas as pd
# %%
df = pd.read_csv('metadata.csv')
df.lesion_type[df.lesion_type.isna()] = 'None'
df['filename'] = df.file.apply(lambda o: o.split('images/')[1])
df.pop('file')
# %%
data = dict(title='Synthetic CT Datasets of Intracranial Hemorrhage',
            columns=[o for o in df.columns if o not in ['filename']],
            references=[
                dict(title="[code]", url='https://github.com/DIDSR/PedSilicoICH')
                ],
            images=[])

for idx, row in df.iterrows():
    data['images'].append(row.to_dict())

with open("metadata.json", "w") as outfile:
    json.dump(data, outfile, indent=True)
# %%
