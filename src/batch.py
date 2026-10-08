import time
import numpy as np


def run_benchmark(jets: list, batch_size: int, inferencer) -> dict:
    """
    Process jets in fixed-size chunks and measure throughput statistics.
    """

    chunk_latencies = []
    all_results = []

    for i in range(0, len(jets), batch_size):
        chunk = jets[i : i + batch_size]

        t0 = time.perf_counter()
        batch_result = inferencer.predict_batch(chunk)
        chunk_latencies.append((time.perf_counter() - t0) * 1000)

        all_results.extend(batch_result['results'])

    latencies = np.array(chunk_latencies)
    total_ms = float(latencies.sum())

    return {
    'results':           all_results,
    'n_jets':            len(jets),
    'batch_size':        batch_size,
    'p50_chunk_ms':      float(np.percentile(latencies, 50)),
    'p95_chunk_ms':      float(np.percentile(latencies, 95)),
    'p99_chunk_ms':      float(np.percentile(latencies, 99)),
    'throughput_jets_s': len(jets) / (total_ms / 1000),
    'total_ms':          total_ms,
}