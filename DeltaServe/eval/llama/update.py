import pandas as pd
import numpy as np

def add_offset_to_column(csv_in, csv_out, column, offset):
    """
    Load a CSV, add an offset to a selected column, and write back to file.

    Parameters
    ----------
    csv_in : str
        Path to input CSV.
    csv_out : str
        Output path for modified CSV. Can be same as csv_in.
    column : str
        Name of column to offset.
    offset : float or int
        Value to add to the column.
    """

    df = pd.read_csv(csv_in)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV.")

    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' is not numeric and cannot be offset.")

    # add offset safely
    col = df[column].astype(float)
    df[column] = np.where(col != 0, col + float(offset), col)
    # when saved, make definition about how many digit to keep:
    df[column] = df[column].round(6)
    df.to_csv(csv_out, index=False)
    print(f"✅ Offset added to '{column}', saved → {csv_out}")
    return df

if __name__ == "__main__":
    csv_in = "batch_prediction_stats.csv"
    csv_out = "batch_prediction_stats.csv"
    column = "predicted_duration"
    offset = -0.001
    add_offset_to_column(csv_in, csv_out, column, offset)