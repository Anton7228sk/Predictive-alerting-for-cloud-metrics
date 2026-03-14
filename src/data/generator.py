import numpy as np
import pandas as pd
from typing import Optional


def _generate_base_metrics(n_points: int, rng: np.random.Generator) -> pd.DataFrame:
    t = np.linspace(0, 4 * np.pi, n_points)

    cpu = (
        30
        + 10 * np.sin(t / (24 * 60) * 2 * np.pi)
        + 5 * np.sin(t / (7 * 24 * 60) * 2 * np.pi)
        + rng.normal(0, 3, n_points)
    )

    memory = (
        55
        + 8 * np.sin(t / (24 * 60) * 2 * np.pi + np.pi / 4)
        + rng.normal(0, 2, n_points)
    )

    latency = np.exp(
        3.5
        + 0.3 * np.sin(t / (24 * 60) * 2 * np.pi)
        + rng.normal(0, 0.2, n_points)
    )

    error_rate = np.clip(
        0.5 + 0.3 * np.abs(rng.normal(0, 1, n_points)), 0, 10
    )

    network_in = np.clip(
        500
        + 200 * np.sin(t / (24 * 60) * 2 * np.pi)
        + rng.normal(0, 50, n_points),
        0,
        None,
    )

    network_out = np.clip(
        300
        + 150 * np.sin(t / (24 * 60) * 2 * np.pi + np.pi / 6)
        + rng.normal(0, 40, n_points),
        0,
        None,
    )

    return pd.DataFrame(
        {
            "cpu_utilization": cpu,
            "memory_usage": memory,
            "request_latency": latency,
            "error_rate": error_rate,
            "network_in": network_in,
            "network_out": network_out,
        }
    )


def _inject_incidents(
    df: pd.DataFrame,
    incident_rate: float,
    rng: np.random.Generator,
    freq_minutes: int,
    pre_ramp_minutes: int = 10,
) -> np.ndarray:
    n_points = len(df)
    incident_flags = np.zeros(n_points, dtype=bool)

    min_gap = int(120 / freq_minutes)
    min_duration = int(10 / freq_minutes)
    max_duration = int(60 / freq_minutes)
    pre_ramp_steps = int(pre_ramp_minutes / freq_minutes)

    i = min_gap
    while i < n_points - max_duration:
        if rng.random() < incident_rate * freq_minutes / 60:
            duration = rng.integers(min_duration, max_duration + 1)
            end = min(i + duration, n_points)
            severity = rng.uniform(0.5, 1.5)

            pre_start = max(0, i - pre_ramp_steps)
            pre_ramp = np.linspace(0, 0.4, i - pre_start)
            for k, j in enumerate(range(pre_start, i)):
                f = pre_ramp[k]
                df.loc[j, "cpu_utilization"] += f * severity * 30
                df.loc[j, "request_latency"] *= 1 + f * severity * 1.5
                df.loc[j, "error_rate"] += f * severity * 3

            ramp_len = min(int(duration * 0.3) + 1, duration)
            ramp = np.linspace(0.4, 1.0, ramp_len)
            for j in range(i, end):
                factor = ramp[j - i] if j - i < ramp_len else 1.0
                df.loc[j, "cpu_utilization"] += factor * severity * rng.uniform(20, 50)
                df.loc[j, "memory_usage"] += factor * severity * rng.uniform(15, 35)
                df.loc[j, "request_latency"] *= 1 + factor * severity * rng.uniform(2, 8)
                df.loc[j, "error_rate"] += factor * severity * rng.uniform(5, 20)
                df.loc[j, "network_in"] += factor * severity * rng.uniform(100, 500)

            incident_flags[i:end] = True
            i = end + min_gap
        else:
            i += 1

    return incident_flags


def generate_dataset(
    n_days: int = 90,
    freq_minutes: int = 1,
    incident_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_points = n_days * 24 * 60 // freq_minutes

    timestamps = pd.date_range(
        start="2024-01-01", periods=n_points, freq=f"{freq_minutes}min"
    )

    df = _generate_base_metrics(n_points, rng)
    incident_flags = _inject_incidents(df, incident_rate, rng, freq_minutes)

    df["cpu_utilization"] = np.clip(df["cpu_utilization"], 0, 100)
    df["memory_usage"] = np.clip(df["memory_usage"], 0, 100)
    df["request_latency"] = np.clip(df["request_latency"], 0, None)
    df["error_rate"] = np.clip(df["error_rate"], 0, 100)
    df["network_in"] = np.clip(df["network_in"], 0, None)
    df["network_out"] = np.clip(df["network_out"], 0, None)

    df.index = timestamps
    df.index.name = "timestamp"
    df["incident"] = incident_flags

    return df
