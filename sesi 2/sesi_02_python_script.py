import multiprocessing as mp
import numpy as np
import time

def monte_carlo_pi_chunk(n_samples):
    np.random.seed()
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    inside_circle = np.sum(x**2 + y**2 <= 1.0)
    return inside_circle

def run_parallel_simulation(total_samples=10_000_000):
    num_workers = mp.cpu_count()
    print(f"=== [SESI 02] HPC Monte Carlo Pi Simulation ===")
    print(f"[+] Total Sampel: {total_samples:,}")
    print(f"[+] Jumlah Worker CPU Tersedia: {num_workers}")
    
    samples_per_worker = total_samples // num_workers
    chunks = [samples_per_worker] * num_workers
    
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(monte_carlo_pi_chunk, chunks)
        
    total_inside = sum(results)
    pi_estimate = 4 * total_inside / total_samples
    elapsed_time = time.time() - start_time
    
    print(f"[+] Estimasi nilai Pi : {pi_estimate:.6f}")
    print(f"[+] Waktu Eksekusi   : {elapsed_time:.4f} detik")

if __name__ == '__main__':
    run_parallel_simulation()