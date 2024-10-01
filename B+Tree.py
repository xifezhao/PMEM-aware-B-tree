import threading
import numpy as np
import sys
import pickle
import zlib
import time
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Global configurations
CACHE_LINE_SIZE = 512  # Simulated cache line size
BATCH_SIZE = 10       # Write batch size
LOG_FILE = 'persistent_log.bin'  # Persistent log file

# Data structure module

class BPlusTreeNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []  # For leaf nodes, values are actual values; for internal nodes, values are child node references
        self.next = None  # For the linked list structure of leaf nodes
        self.lock = threading.Lock()
        self.version = 0  # For optimistic locking

    def size_in_bytes(self):
        # Convert list to NumPy array to simulate compact array
        keys_array = np.array(self.keys, dtype=np.int64)
        values_array = np.array(self.values, dtype=object)
        # Calculate the byte size of the node
        header_size = sys.getsizeof(self.is_leaf) + sys.getsizeof(self.next) + sys.getsizeof(self.version)
        keys_size = keys_array.nbytes
        values_size = values_array.nbytes
        total_size = header_size + keys_size + values_size
        return total_size

    def is_cache_aligned(self):
        # Check if node size is a multiple of cache line size
        return self.size_in_bytes() % CACHE_LINE_SIZE == 0

    def insert(self, key, value):
        # Simplified insertion method, does not include balancing
        self.keys.append(key)
        self.values.append(value)
        # Update version number
        self.version += 1

    def __str__(self):
        return f"Node(is_leaf={self.is_leaf}, keys={self.keys}, values={self.values})"

# Write optimization module

class LogStructuredStore:
    def __init__(self, batch_size=BATCH_SIZE):
        self.log_buffer = []
        self.batch_size = batch_size
        self.persistent_log = []
        self.lock = threading.Lock()
        self.log_file = LOG_FILE
        # Load persistent log
        self.load_persistent_log()

    def compress_data(self, data):
        serialized_data = pickle.dumps(data)
        compressed_data = zlib.compress(serialized_data)
        return compressed_data

    def decompress_data(self, compressed_data):
        decompressed_data = zlib.decompress(compressed_data)
        data = pickle.loads(decompressed_data)
        return data

    def write(self, update):
        with self.lock:
            self.log_buffer.append(update)
            if len(self.log_buffer) >= self.batch_size:
                self.flush()

    def flush(self):
        # Compress batch updates
        compressed_batch = self.compress_data(self.log_buffer)
        # Get the length of compressed data
        batch_length = len(compressed_batch)
        # Write length to file (using 4-byte unsigned integer, big-endian)
        with open(self.log_file, 'ab') as f:
            f.write(batch_length.to_bytes(4, byteorder='big'))
            f.write(compressed_batch)
        self.persistent_log.append(compressed_batch)
        # Clear log buffer
        self.log_buffer = []

    def load_persistent_log(self):
        # Load persistent log
        if os.path.exists(self.log_file):
            with open(self.log_file, 'rb') as f:
                while True:
                    # Read 4-byte data length
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break
                    if len(length_bytes) < 4:
                        print("Warning: Incomplete data detected in log file (length bytes).")
                        break
                    batch_length = int.from_bytes(length_bytes, byteorder='big')
                    # Read compressed data based on read length
                    compressed_batch = f.read(batch_length)
                    if len(compressed_batch) != batch_length:
                        # Data is incomplete, possibly due to file corruption, skip this data block
                        print("Warning: Incomplete data detected in log file (data block).")
                        continue  # Skip current loop, continue reading next data block
                    self.persistent_log.append(compressed_batch)

    def read_persistent_log(self):
        # Read and decompress all entries in the persistent log
        all_updates = []
        for compressed_batch in self.persistent_log:
            batch_updates = self.decompress_data(compressed_batch)
            all_updates.extend(batch_updates)
        return all_updates

    def clear_log(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.persistent_log = []
        self.log_buffer = []

# Concurrency control module

class ConcurrentBPlusTree:
    def __init__(self):
        self.root = BPlusTreeNode(is_leaf=True)
        self.store = LogStructuredStore()
        self.lock = threading.Lock()

    def insert(self, key, value):
        # Pre-write log
        update = {'key': key, 'value': value}
        self.store.write(update)
        # Optimistic locking
        while True:
            version_before = self.root.version
            with self.root.lock:
                self.root.insert(key, value)
            version_after = self.root.version
            if version_after == version_before + 1:
                break
            else:
                print(f"Version conflict, retrying insertion for key {key}")
        # If batch size is reached, flush log
        if len(self.store.log_buffer) >= BATCH_SIZE:
            self.store.flush()

    def apply_update(self, update):
        with self.root.lock:
            self.root.insert(update['key'], update['value'])

    def recover(self):
        # Recover from persistent log
        updates = self.store.read_persistent_log()
        for update in updates:
            self.apply_update(update)

    def clear_logs(self):
        self.store.clear_log()

# Benchmarking module

class Benchmark:
    def __init__(self, tree, num_operations=1000, num_threads=4):
        self.tree = tree
        self.num_operations = num_operations
        self.num_threads = num_threads
        self.insert_times = []
        self.lock = threading.Lock()

    def worker(self, start_key):
        times = []
        for i in range(self.num_operations):
            key = start_key + i
            value = key * 10
            start_time = time.time()
            self.tree.insert(key, value)
            end_time = time.time()
            times.append(end_time - start_time)
        with self.lock:
            self.insert_times.extend(times)

    def run(self):
        threads = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i in range(self.num_threads):
                executor.submit(self.worker, i * self.num_operations)
        # Wait for all threads to complete
        time.sleep(1)  # Wait for all threads to complete insertion
        # Flush all remaining logs
        self.tree.store.flush()

    def visualize_results(self):
        # Setting support for Chinese fonts
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # Using SimHei font
        plt.rcParams['axes.unicode_minus'] = False  # Solving the problem of negative numbers display in axis

        # Plotting histogram of insertion times
        plt.hist(self.insert_times, bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of insertion operation times')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.show()

        # Calculate average insertion time
        average_time = sum(self.insert_times) / len(self.insert_times)
        print(f"Average time per insertion operation: {average_time * 1000:.4f} milliseconds")

# Main program

if __name__ == "__main__":
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    # Create PMEM-aware B+ tree
    tree = ConcurrentBPlusTree()
    # Benchmark parameters
    NUM_OPERATIONS = 1000
    NUM_THREADS = 4

    # Create benchmark instance
    benchmark = Benchmark(tree, num_operations=NUM_OPERATIONS, num_threads=NUM_THREADS)

    print("Starting benchmark...")
    start_time = time.time()
    benchmark.run()
    end_time = time.time()
    print(f"Benchmark completed, total time taken: {end_time - start_time:.2f} seconds")

    # Visualize results
    benchmark.visualize_results()

    # Testing crash recovery
    print("Simulating system crash and recovery...")
    # Recreate tree, simulating system reboot
    tree_recovered = ConcurrentBPlusTree()
    tree_recovered.recover()
    print("Recovery complete.")
    print(f"Recovered root node: {tree_recovered.root}")
    print(f"Node size after recovery (bytes): {tree_recovered.root.size_in_bytes()}")
    print(f"Is cache line aligned: {tree_recovered.root.is_cache_aligned()}")

    # Clear log file
    tree_recovered.clear_logs()
