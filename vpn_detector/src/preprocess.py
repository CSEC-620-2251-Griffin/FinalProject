import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_data_files(raw_dir: Path) -> List[Path]:
    patterns = ["**/*.pcap", "**/*.pcapng", "**/*.csv", "**/*.hdf5", "**/*.h5"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(raw_dir.glob(pat))
    return sorted(set(files))


def tshark_to_csv(pcap_path: Path, output_csv: Path) -> None:
    cmd = [
        "tshark",
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-e",
        "frame.time_delta",
        "-e",
        "frame.len",
        "-e",
        "ip.proto",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "tls.record.content_type",
        "-E",
        "header=y",
        "-E",
        "separator=,",
    ]
    with output_csv.open("w") as f:
        subprocess.run(cmd, check=True, stdout=f)


def entropy(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    values, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "frame.time_delta": "time_delta",
        "frame.time_relative": "time_relative",
        "frame.time_epoch": "time_epoch",
        "frame.len": "length",
        "ip.proto": "proto",
        "ip.src": "src_ip",
        "ip.dst": "dst_ip",
        "tcp.srcport": "tcp_srcport",
        "tcp.dstport": "tcp_dstport",
        "udp.srcport": "udp_srcport",
        "udp.dstport": "udp_dstport",
        "tls.record.content_type": "tls_content_type",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


def infer_timestamp(df: pd.DataFrame) -> pd.Series:
    if "time_epoch" in df.columns:
        return pd.to_numeric(df["time_epoch"], errors="coerce")
    if "time_relative" in df.columns:
        return pd.to_numeric(df["time_relative"], errors="coerce")
    if "time_delta" in df.columns:
        time_delta = pd.to_numeric(df["time_delta"], errors="coerce").fillna(0.0)
        return time_delta.cumsum()
    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)


def prepare_packet_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / f"{path.stem}.csv"
            tshark_to_csv(path, csv_path)
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    elif suffix in {".csv"}:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    elif suffix in {".hdf5", ".h5"}:
        # Try default key then fall back to first available
        try:
            df = pd.read_hdf(path)
        except (KeyError, ValueError):
            with pd.HDFStore(path, mode="r") as store:
                first_key = store.keys()[0]
            df = pd.read_hdf(path, key=first_key)
    else:
        raise ValueError(f"Unsupported format: {path}")
    df = normalize_columns(df)
    # Ensure expected columns exist even if empty (prevents downstream KeyErrors)
    expected_cols = [
        "src_ip",
        "dst_ip",
        "tcp_srcport",
        "tcp_dstport",
        "udp_srcport",
        "udp_dstport",
        "proto",
        "length",
        "tls_content_type",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    df["timestamp"] = infer_timestamp(df)
    return df


def extract_ports(df: pd.DataFrame) -> pd.DataFrame:
    df["src_port"] = (
        pd.to_numeric(df.get("tcp_srcport"), errors="coerce")
        .fillna(pd.to_numeric(df.get("udp_srcport"), errors="coerce"))
        .fillna(-1)
        .astype(int)
    )
    df["dst_port"] = (
        pd.to_numeric(df.get("tcp_dstport"), errors="coerce")
        .fillna(pd.to_numeric(df.get("udp_dstport"), errors="coerce"))
        .fillna(-1)
        .astype(int)
    )
    return df


def aggregate_flows(df: pd.DataFrame, file_path: Path, label: int, capture_id: str) -> pd.DataFrame:
    required_cols = {"src_ip", "dst_ip"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        print(f"[preprocess] Missing required columns {missing} in {file_path}, skipping capture")
        return pd.DataFrame()
    df = extract_ports(df)
    df["proto"] = pd.to_numeric(df.get("proto"), errors="coerce").fillna(-1).astype(int)
    df["length"] = pd.to_numeric(df.get("length"), errors="coerce").fillna(0).astype(float)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["src_ip", "dst_ip"])
    if df.empty:
        print(f"[preprocess] No IP packets in {file_path}, skipping capture")
        return pd.DataFrame()

    df["tls_flag"] = (~df.get("tls_content_type", pd.Series(index=df.index)).isna()).astype(int)

    group_cols = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
    grouped = df.groupby(group_cols)

    records: List[Dict] = []
    for idx, (key, g) in enumerate(grouped):
        g = g.sort_values("timestamp")
        pkt_lengths = g["length"].values
        timestamps = g["timestamp"].values
        inter_arrival = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
        duration = float(timestamps.max() - timestamps.min()) if len(timestamps) > 0 else 0.0
        duration = duration if duration > 0 else 1e-6
        bytes_up = g.loc[g["src_ip"] == key[0], "length"].sum()
        bytes_down = g.loc[g["dst_ip"] == key[2], "length"].sum()
        record = {
            "flow_id": f"{capture_id}_{idx}",
            "src_ip": key[0],
            "dst_ip": key[2],
            "src_port": key[1],
            "dst_port": key[3],
            "proto": key[4],
            "pkt_count": len(g),
            "bytes_total": g["length"].sum(),
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "flow_duration": duration,
            "pkt_len_mean": float(np.mean(pkt_lengths)),
            "pkt_len_std": float(np.std(pkt_lengths)),
            "pkt_len_min": float(np.min(pkt_lengths)),
            "pkt_len_max": float(np.max(pkt_lengths)),
            "iat_mean": float(np.mean(inter_arrival)),
            "iat_std": float(np.std(inter_arrival)),
            "iat_min": float(np.min(inter_arrival)),
            "iat_max": float(np.max(inter_arrival)),
            "entropy_packet_sizes": entropy(pkt_lengths),
            "pkt_rate": len(g) / duration,
            "byte_rate": g["length"].sum() / duration,
            "direction_ratio": bytes_up / (bytes_down + 1e-6),
            "tls_record_count": int(g["tls_flag"].sum()),
            "label": label,
            "capture_id": capture_id,
            "file_name": file_path.name,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def derive_label_from_name(path: Path) -> Tuple[int, str]:
    name = path.name.lower()
    if "nonvpn" in name or "non-vpn" in name:
        label = 0
    elif "vpn" in name:
        label = 1
    else:
        label = 0
    return label, path.stem


def preprocess(config: Dict) -> Path:
    raw_dir = Path(config["data"]["raw_dir"])
    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    files = find_data_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No data files found under {raw_dir}")

    print(f"[preprocess] Found {len(files)} files. Converting to flows...")
    all_flows: List[pd.DataFrame] = []
    for path in tqdm(files, desc="Processing captures"):
        label, capture_id = derive_label_from_name(path)
        try:
            pkt_df = prepare_packet_df(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[preprocess] Skipping {path} due to error: {exc}")
            continue
        flow_df = aggregate_flows(pkt_df, path, label=label, capture_id=capture_id)
        if not flow_df.empty:
            all_flows.append(flow_df)

    if not all_flows:
        raise RuntimeError("No flows generated from the provided captures.")

    combined = pd.concat(all_flows, ignore_index=True)
    combined.to_parquet(processed_path, index=False)
    meta = {
        "source_dir": str(raw_dir),
        "files_processed": len(all_flows),
        "num_rows": len(combined),
        "num_features": len(combined.columns),
    }
    meta_path = processed_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] Saved flows to {processed_path} ({len(combined)} rows)")
    return processed_path


def main(config_path: str) -> None:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    preprocess(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess VNAT captures into flow-level parquet.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
