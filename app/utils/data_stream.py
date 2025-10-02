import pandas as pd
import random
import time

def stream_data(file_path, chunksize=1000):
    for chunk in pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk = chunk.dropna()
        numeric_cols = [
            "Global_active_power", "Global_reactive_power",
            "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
        ]
        chunk[numeric_cols] = chunk[numeric_cols].astype(float)
        yield chunk

def simulate_live_stream(file_path, rate=1.0):
    for batch in stream_data(file_path, chunksize=500):
        for _, row in batch.iterrows():
            row["Global_active_power"] += random.uniform(-0.1, 0.1)
            row["Voltage"] += random.uniform(-0.5, 0.5)
            yield {
                "timestamp": row["datetime"],
                "power": row["Global_active_power"],
                "voltage": row["Voltage"],
                "intensity": row["Global_intensity"],
                "sub_metering": [
                    row["Sub_metering_1"],
                    row["Sub_metering_2"],
                    row["Sub_metering_3"]
                ]
            }
            time.sleep(rate)
