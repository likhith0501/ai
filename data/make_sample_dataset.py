"""Generate a reproducible synthetic unidirectional IP traffic dataset.

The generator is only a convenience for local development and demonstration.
Replace ``data/traffic_dataset.csv`` with a real capture (for example an
exported CICIDS2017, UNSW-NB15 or CSE-CIC-IDS2018 flow table) and the whole
pipeline keeps working: column names are mapped automatically.

Usage:  python data/make_sample_dataset.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

RANDOM_STATE = 42
NORMAL = "Normal"

CLASS_SPECS = [
    ("Normal", 4200),
    ("DoS", 900),
    ("DDoS", 850),
    ("Port Scan", 1150),
    ("Brute Force", 800),
    ("Botnet", 750),
    ("Infiltration", 650),
    ("Malware", 700),
]

INTERNAL_NETS = ["192.168.10", "192.168.20", "10.0.0", "172.16.4"]
EXTERNAL_NETS = ["203.0.113", "198.51.100", "45.77.9", "104.21.6"]
SERVICE_PORTS = [80, 443, 53, 22, 23, 3389, 445, 8080, 25, 110, 21, 3306, 1433, 5900]
FLAGS = ["S", "SA", "A", "FA", "PA", "ACK", "SYN", "PSH", "FIN", "RST", "0x002", "0x010"]


def _ip(rng: np.random.Generator, internal: bool) -> str:
    nets = INTERNAL_NETS if internal else EXTERNAL_NETS
    return f"{nets[rng.integers(len(nets))]}.{rng.integers(1, 254)}"


def _sample_class(rng: np.random.Generator, label: str, size: int) -> pd.DataFrame:
    proto_pick = rng.random(size)
    tcp = proto_pick < 0.72
    udp = (proto_pick >= 0.72) & (proto_pick < 0.93)
    icmp = proto_pick >= 0.93

    internal_src = rng.random(size) < 0.75
    src_ip = [_ip(rng, bool(internal_src[i])) for i in range(size)]
    dst_ip = [_ip(rng, not internal_src[i]) for i in range(size)]

    data: dict = {
        "Flow_ID": [f"flow-{i:06d}" for i in range(size)],
        "Source_IP": src_ip,
        "Destination_IP": dst_ip,
        "Source_Port": rng.integers(32768, 60999, size),
        "Destination_Port": rng.choice(SERVICE_PORTS, size),
        "Protocol": np.where(tcp, "TCP", np.where(udp, "UDP", "ICMP")),
        "Protocol_Type": np.where(tcp, 6, np.where(udp, 17, 1)),
    }

    if label == NORMAL:
        packets = rng.integers(8, 260, size)
        bytes_total = packets * rng.integers(180, 1400, size) + rng.integers(0, 40000, size)
        duration = np.abs(rng.normal(2.4, 1.6, size)) + 0.02
        fwd_ratio = rng.uniform(0.3, 0.7, size)
        bytes_ratio = rng.uniform(0.25, 0.75, size)
        iat = np.abs(rng.normal(0.004, 0.003, size))
        data["TCP_Flags"] = rng.choice(FLAGS[1:8], size)

    elif label == "DoS":
        packets = rng.integers(900, 6500, size)
        bytes_total = packets * rng.integers(48, 130, size)
        duration = np.abs(rng.normal(1.6, 0.9, size)) + 0.05
        fwd_ratio = rng.uniform(0.97, 1.0, size)
        bytes_ratio = rng.uniform(0.95, 1.0, size)
        iat = np.abs(rng.normal(0.0004, 0.0003, size))
        data["TCP_Flags"] = rng.choice(["S", "SA", "0x002"], size)

    elif label == "DDoS":
        packets = rng.integers(2500, 16000, size)
        bytes_total = packets * rng.integers(60, 200, size)
        duration = np.abs(rng.normal(3.1, 1.9, size)) + 0.05
        fwd_ratio = rng.uniform(0.98, 1.0, size)
        bytes_ratio = rng.uniform(0.97, 1.0, size)
        iat = np.abs(rng.normal(0.0002, 0.00015, size))
        data["TCP_Flags"] = rng.choice(["0x002", "S", "0x018"], size)

    elif label == "Port Scan":
        packets = rng.integers(1, 9, size)
        bytes_total = packets * rng.integers(40, 220, size)
        duration = np.abs(rng.normal(0.35, 0.3, size)) + 0.001
        fwd_ratio = rng.uniform(0.99, 1.0, size)
        bytes_ratio = rng.uniform(0.99, 1.0, size)
        iat = np.abs(rng.normal(0.02, 0.015, size))
        data["Destination_Port"] = rng.integers(1024, 65000, size)
        data["TCP_Flags"] = rng.choice(["S", "0x002"], size)

    elif label == "Brute Force":
        packets = rng.integers(120, 1400, size)
        bytes_total = packets * rng.integers(120, 900, size)
        duration = np.abs(rng.normal(9.5, 5.0, size)) + 0.5
        fwd_ratio = rng.uniform(0.55, 0.8, size)
        bytes_ratio = rng.uniform(0.35, 0.65, size)
        iat = np.abs(rng.normal(0.0009, 0.0005, size))
        data["Destination_Port"] = rng.choice([22, 23, 3389, 21, 5900], size)
        data["TCP_Flags"] = rng.choice(["SA", "PA", "PSH", "A"], size)

    elif label == "Botnet":
        packets = rng.integers(40, 900, size)
        bytes_total = packets * rng.integers(200, 1100, size)
        duration = np.abs(rng.normal(60.0, 25.0, size)) + 5.0
        fwd_ratio = rng.uniform(0.2, 0.5, size)
        bytes_ratio = rng.uniform(0.15, 0.45, size)
        iat = np.abs(rng.normal(30.0, 6.0, size))
        data["Destination_Port"] = rng.integers(1024, 65000, size)
        data["TCP_Flags"] = rng.choice(["ACK", "PSH", "A"], size)

    elif label == "Infiltration":
        packets = rng.integers(25, 300, size)
        bytes_total = packets * rng.integers(300, 1500, size)
        duration = np.abs(rng.normal(120.0, 45.0, size)) + 10.0
        fwd_ratio = rng.uniform(0.12, 0.35, size)
        bytes_ratio = rng.uniform(0.1, 0.3, size)
        iat = np.abs(rng.normal(4.0, 2.0, size))
        data["Destination_Port"] = rng.integers(4000, 65000, size)
        data["TCP_Flags"] = rng.choice(["PA", "PSH", "A"], size)

    else:
        label = "Malware"
        packets = rng.integers(150, 3200, size)
        bytes_total = packets * rng.integers(400, 2100, size)
        duration = np.abs(rng.normal(14.0, 8.0, size)) + 0.3
        fwd_ratio = rng.uniform(0.45, 0.95, size)
        bytes_ratio = rng.uniform(0.4, 0.95, size)
        iat = np.abs(rng.normal(0.002, 0.0015, size))
        data["TCP_Flags"] = rng.choice(["PSH", "FA", "A", "PA"], size)

    forward_packets = np.maximum(1, (packets * fwd_ratio).astype(int))
    reverse_packets = np.clip(packets - forward_packets, 0, None)
    forward_bytes = np.maximum(1, (bytes_total * bytes_ratio).astype(int))
    reverse_bytes = np.clip(bytes_total - forward_bytes, 0, None)

    data["Packet_Count"] = packets
    data["Byte_Count"] = bytes_total
    data["Packet_Length"] = np.where(
        packets > 0, np.maximum(1, bytes_total // np.maximum(packets, 1)), 0
    )
    data["Flow_Duration"] = (duration * 1_000_000).astype(int)
    data["Forward_Packets"] = forward_packets
    data["Forward_Bytes"] = forward_bytes
    data["Reverse_Packets"] = reverse_packets
    data["Reverse_Bytes"] = reverse_bytes
    data["TCP_Flags"] = data["TCP_Flags"]
    data["Inter_Arrival_Time"] = np.round(iat, 6)
    data["Label"] = label

    frame = pd.DataFrame(data)
    frame["Protocol"] = frame["Protocol"].where(~icmp, "ICMP")
    frame["Source_Port"] = frame["Source_Port"].astype(int)
    return frame


def _blend_features(frame: pd.DataFrame, targets: np.ndarray, sources: np.ndarray,
                    columns: List[str], rng: np.random.Generator, ratio: float) -> None:
    """Copy feature values between rows of ``targets`` and randomly chosen ``sources``.

    This creates genuine class overlap so the reported metrics are not perfect.
    """
    count = int(len(targets) * ratio)
    if count <= 0 or len(sources) == 0:
        return
    chosen = rng.choice(targets, size=count, replace=False)
    donors = sources[rng.integers(0, len(sources), size=count)]
    for column in columns:
        location = frame.columns.get_loc(column)
        frame.iloc[chosen, location] = frame.iloc[donors, location].to_numpy()


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    frames = [_sample_class(rng, label, size) for label, size in CLASS_SPECS]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    frame["Flow_ID"] = [f"flow-{i:06d}" for i in range(len(frame))]

    numeric_columns = [
        "Source_Port", "Destination_Port", "Packet_Count", "Byte_Count", "Packet_Length",
        "Flow_Duration", "Forward_Packets", "Forward_Bytes", "Reverse_Packets",
        "Reverse_Bytes", "Inter_Arrival_Time",
    ]
    frame[numeric_columns] = frame[numeric_columns].astype("float64")

    labels = frame["Label"].to_numpy()
    unidirectional = rng.random(len(frame)) < 0.14
    frame.loc[unidirectional, "Reverse_Packets"] = 0.0
    frame.loc[unidirectional, "Reverse_Bytes"] = 0.0

    noisy = rng.random(len(frame)) < 0.025
    noisy_indices = np.flatnonzero(noisy)
    for index in noisy_indices:
        values = frame.loc[index, numeric_columns].to_numpy(dtype=float)
        noise = rng.normal(0.0, 1.0, values.shape)
        scales = np.maximum(np.abs(values), 1.0)
        frame.loc[index, numeric_columns] = np.maximum(values + noise * scales * 0.6, 0.0)
        alternatives = [label for label, _ in CLASS_SPECS if label != labels[index]]
        frame.loc[index, "Label"] = str(rng.choice(alternatives))

    overlap_ratio = 0.12
    donor_columns = numeric_columns + ["Destination_Port", "Protocol", "Protocol_Type", "TCP_Flags"]
    labels = frame["Label"].to_numpy()
    normal_positions = np.flatnonzero(labels == NORMAL)
    attack_positions = np.flatnonzero(labels != NORMAL)

    _blend_features(frame, attack_positions, normal_positions, donor_columns, rng, overlap_ratio)
    _blend_features(frame, normal_positions, attack_positions, donor_columns, rng, overlap_ratio * 0.6)

    dirty_mask = rng.random(len(frame)) < 0.012
    frame.loc[dirty_mask, "Inter_Arrival_Time"] = np.nan
    frame.loc[dirty_mask, "Packet_Length"] = np.nan
    frame.loc[rng.choice(len(frame), 4, replace=False), "Byte_Count"] = np.inf
    frame = pd.concat([frame, frame.iloc[[5, 42, 91]]], ignore_index=True)
    return frame


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "traffic_dataset.csv")
    frame = build_dataset()
    frame.to_csv(target, index=False)
    print(f"Written {len(frame)} rows x {frame.shape[1]} columns to {target}")
    print(frame["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()