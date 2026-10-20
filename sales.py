# # import glob
# # import os
# # import kagglehub
# # import numpy as np

# # # Download dataset (returns a string path)
# # dataset_path = kagglehub.dataset_download("anairamcosta/sales-csv")
# # print("Path to dataset files:", dataset_path)

# # # Find CSV files inside the downloaded folder
# # csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)

# # if not csv_files:
# #     raise FileNotFoundError(f"No CSV file found in: {dataset_path}")

# # csv_path = csv_files[0]
# # print("Loading:", csv_path)

# # # Load the CSV. Remove skiprows=1 if it has no header row.
# # loaded = np.loadtxt(csv_path, delimiter=",", dtype=str, skiprows=1)

# # # Save example data
# # sales = np.array([[10, 200], [12, 240]])
# # output_path = os.path.join(dataset_path, "sales-output.csv")
# # np.savetxt(output_path, sales, delimiter=",", fmt="%d")

# # print(loaded)
# # print("Saved example data to:", output_path)


# # import kagglehub
# # import numpy as np

# # folder = kagglehub.dataset_download("anairamcosta/sales-csv")
# # print("Dataset folder:", folder)

# # # Change sales.csv to the actual CSV filename in that folder
# # csv_path = folder + "/sales.csv"

# # loaded = np.loadtxt(csv_path, delimiter=",", dtype=str, skiprows=1)

# sales = np.array([[10, 200], [12, 240]])
# # np.savetxt("sales-output.csv", sales, delimiter=",", fmt="%d")

# # print(loaded)



# import pandas as pd
# days=pd.Series(['mon','tue','wed'],
# name="days")
# print(days)


# product=pd.DataFrame({"code":["ai","b1","c1"],"price":[10,20,30]} )
# print(product)


import pandas as pd
data={
    "product":["mouse","chair","table","pen","book","tablet"],
    "quantity":[2,1,3,4,5,None],
    "unitsold":[1,2,3,4,5,6]
}

df=pd.DataFrame(data)

print("Real data")
print(df)


print("\n Numeric Summary\n")
print(df.describe())

print("\n Catalitecal Discriptiom\n")
print(df.describe(include=["object"]))

print("\nMissing vallue\n")
print(df.isna().sum())
