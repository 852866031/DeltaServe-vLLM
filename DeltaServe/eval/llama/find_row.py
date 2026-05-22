import pandas as pd

def find_rows_below_threshold(csv_path, column_name, threshold):
    df = pd.read_csv(csv_path)
    idx = df.index[df[column_name] < threshold].tolist()
    return idx

path = "results/latency_co-serving.csv"
entry = "latency_s"
rows = find_rows_below_threshold(path, entry, 1)
print(rows)