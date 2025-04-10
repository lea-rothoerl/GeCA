import pandas as pd

df1 = pd.read_csv("breast-level_annotations.csv")
df2 = pd.read_csv("metadata.csv")
df3 = pd.read_csv("finding_annotations.csv")
df3['finding_idx'] = df3.index

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df3.columns = df3.columns.str.strip()

# column selection 
df1 = df1[["image_id", "laterality", "view_position", "height", "width", "split", "breast_density"]]
df2 = df2[["SOP Instance UID", "Manufacturer's Model Name"]] 
df3 = df3[["image_id", "finding_categories", "xmin", "ymin", "xmax", "ymax", "finding_idx"]] 

df_merged = df3.merge(df2, left_on="image_id", right_on="SOP Instance UID", how="left").drop(columns=["SOP Instance UID"])

df_merged = df_merged.merge(df1, on="image_id", how="left")

df_merged.rename(columns={"Manufacturer's Model Name": "model"}, inplace=True)
df_merged.to_csv("annotations.csv", index=False)
print("Done!")
