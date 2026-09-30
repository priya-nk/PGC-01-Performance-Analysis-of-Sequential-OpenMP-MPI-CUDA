
# Parallel Matrix Multiplication Performance Benchmarking

## Table of Contents

1. [Objective](objective)
2. [Theoretical Framework & Architecture Comparison](framework--architecture-comparison)
   - [Sequential Computing](sequential-computing)
   - [OpenMP (Shared-Memory Concurrency)](openmp-shared-memory-concurrency)
   - [MPI (Distributed-Memory Message Passing)](mpi-distributed-memory-message-passing)
   - [CUDA (Massively Parallel GPU Acceleration)](cuda-massively-parallel-gpu-acceleration)
   - [Architecture Comparison Matrix](architecture-comparison-matrix)
3. [Workload Specifications](workload-specifications)
   - [Matrix Configuration & Expected Values](matrix-configuration--expected-values)
   - [Computational Complexity](computational-complexity)
   - [OpenMp Configuration](openmp-configuration)
   - [MPI Configuration](mpi-configuration)
   - [CUDA Configuration](cuda-configuration)
   - [Overall Execution Flow](overall-execution-flow)
4. [Build and Execution Protocols](build-and-execution-protocols)
   - [Sequential Implementation](sequential-implementation)
   - [OpenMP Implementation](openmp-implementation)
   - [MPI Implementation](mpi-implementation)
   - [CUDA Implementation](cuda-implementation)
5. [Experimental Results](experimental-results)
   - [Execution Time Measurements](execution-time-measurements)
   - [Correctness Verification](correctness-verification)
   - [Speedup Metrics](speedup-metrics)
6. [Detailed Technical Analysis & Overhead Explanation](detailed-technical-analysis--overhead-explanation)
   - [Sequential Baseline Mechanics](sequential-baseline-mechanics)
   - [OpenMP Mechanics & Cache Interactions](openmp-mechanics--cache-interactions)
   - [MPI Mechanics & Network Bottlenecks](mpi-mechanics--network-bottlenecks)
   - [CUDA Mechanics & Diagnostic Failure Analysis](cuda-mechanics--diagnostic-failure-analysis)
7. [Conclusion](conclusion)
8. [Repository Structure](repository-structure)

## 1. Objective

The primary objective of this experiment is to systematically benchmark, compare, and analyze four fundamental high-performance compute paradigms: **Sequential CPU**, **OpenMP multi-threading**, **MPI distributed processing**, and **CUDA GPU acceleration**.

### Specific Goals

- **Standardized Workload Benchmarking:** Implement and execute an identical  double-precision matrix multiplication workload across all four paradigms to evaluate performance under consistent computational stress.
- **Accuracy Verification:** Validate the numerical precision and correctness of each parallel implementation against a known mathematical outcome.
- **Speedup & Efficiency Metrics:** Measure runtime profiles to compute speedup ratios () and compute resource efficiency across multi-core and multi-node hardware.
- **Architectural Bottleneck Identification:** Analyze hardware-level performance boundaries, including memory bandwidth, cache hit/miss rates, inter-process communication latency, thread overhead, and PCIe host-to-device transfers.

## 2. Theoretical Framework & Architecture Comparison

### Sequential Computing

#### Explanation

Sequential computing serves as the control baseline. Execution proceeds strictly in a single line of control using a single CPU core. It computes matrix multiplication cell-by-cell using standard nested loops.

For two input matrices  and , element computation follows:

```
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}

```
Because execution flows strictly linearly without thread or process concurrency, performance is entirely bounded by single-core CPU ALU throughput, clock speed, and cache latency.

### OpenMP (Shared-Memory Concurrency)

#### Explanation

OpenMP introduces multi-threaded parallelism across CPU cores sharing a common memory space. Instead of running loops sequentially, OpenMP forks a team of threads to process independent loop iterations concurrently.

In this implementation, the outermost loop (`i`) is parallelized across 8 hardware threads, assigning distinct row ranges to each thread while avoiding thread data duplication.

```
                  Shared CPU Memory Space
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
  Thread 0                Thread 1                Thread 2 ... Thread 7
  (Rows 0–499)         (Rows 500–999)         (Rows 1000–1499)

```
Key characteristics:

- Direct access to a shared physical address space with minimal memory overhead.
- Thread creation, scheduling, and synchronization handled transparently via compiler directives (`#pragma omp parallel for`).

### MPI (Distributed-Memory Message Passing)

#### Explanation

MPI (Message Passing Interface) establishes parallelism across separate execution units with isolated address spaces. These can run as separate processes on a single computer or spread across multiple networked machines in a cluster.

Because memory is not shared, processes explicitly transfer data using network messages or inter-process communication (IPC) channels. The workload divides matrix rows among 4 distinct MPI ranks:

```
                      Master Process (Rank 0)
                                 │
                   ┌─────────────┴─────────────┐
                   │    MPI_Scatter / Bcast    │
                   ▼                           ▼
        Rank 0 / 1 / 2 / 3           Worker Nodes (Ranks 1–3)
        (1000 Rows Each)              (Isolated Address Spaces)
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                            MPI_Gather
                                 ▼
                         Assembled Matrix C

```
Execution steps:

1. **`MPI_Bcast`:** Matrix  is copied to all processes.
2. **`MPI_Scatter`:** Slices of Matrix  (1000 rows each) are distributed across processes.
3. **Local Multiplication:** Each process calculates its subset of matrix rows.
4. **`MPI_Gather`:** Partial results are aggregated back to the primary process (Rank 0) to assemble Matrix .

### CUDA (Massively Parallel GPU Acceleration)

#### Explanation

CUDA leverages the Single Instruction, Multiple Threads (SIMT) hardware engine of NVIDIA GPUs. Rather than using a few complex CPU cores, CUDA launches thousands of simpler GPU compute threads in parallel.

Every output element  is mapped directly to a dedicated CUDA thread:

- **Block Dimensions:**  (256 threads per thread block)
- **Grid Dimensions:**  blocks ( total blocks)
- **Total Concurrently Launching Threads:**  threads

```
                          NVIDIA GPU
                              │
                          CUDA Grid
               (250 x 250 Blocks = 62,500 Total)
                              │
       ┌──────────────────────┼──────────────────────┐
    Block 0                Block 1                Block N
  (16x16 Threads)        (16x16 Threads)        (16x16 Threads)

```
Data lifecycle:

1. Allocate Host (CPU) memory and Device (GPU VRAM) memory.
2. Transfer matrices  and  over the PCIe bus to VRAM via `cudaMemcpyHostToDevice`.
3. Launch the GPU kernel across thread blocks.
4. Copy calculated Matrix  back from VRAM to Host RAM via `cudaMemcpyDeviceToHost`.

### Architecture Comparison Matrix

| Architectural Feature  | Sequential CPU       | OpenMP                         | MPI                           | CUDA GPU                      |
| ---------------------- | -------------------- | ------------------------------ | ----------------------------- | ----------------------------- |
| **Compute Model**      | Single-threaded      | Shared-Memory                  | Distributed-Memory            | SIMT Massively Parallel       |
| **Hardware Used**      | 1 CPU Core           | Multi-Core CPU                 | Distributed Clusters / CPUs   | Hardware Accelerators (GPU)   |
| **Execution Units**    | 1 Instruction Thread | 8 CPU Threads                  | 4 Operating System Processes  |  Thread Mapping               |
| **Memory Boundaries**  | Local CPU Stack/Heap | Single Shared System RAM       | Isolated Memory per Process   | Separate VRAM (Device) Memory |
| **Inter-Unit Comm.**   | None                 | Shared Pointer Access          | TCP/IP or IPC Message Passing | PCIe Host-to-Device Transfers |
| **Primary Bottleneck** | Raw ALU throughput   | Core contention / Cache misses | Network transmission latency  | PCIe transfer bandwidth       |

## 3. Workload Specifications

The same workload is used for all four implementations to make the performance comparison consistent.

### 3.1 Matrix Configuration & Expected Values

| Matrix       | Dimensions  | Initialization  |
| ------------ | ----------- | --------------- |
| **Matrix A** | 4000 × 4000 | `A[i][j] = 1.0` |
| **Matrix B** | 4000 × 4000 | `B[i][j] = 1.0` |
| **Matrix C** | 4000 × 4000 | `C[i][j] = 0.0` |

For every element of the result matrix:

```text
C[i][j] = A[i][0] × B[0][j]
        + A[i][1] × B[1][j]
        + ...
        + A[i][3999] × B[3999][j]
```

Since every input element is `1.0` and there are 4000 terms:

```text
C[i][j] = 4000.00
```

Therefore, the expected verification value is:

```text
C[0][0] = 4000.00
```

This known expected value is used to check the correctness of the implementations.

### 3.2 Computational Complexity

Standard matrix multiplication uses three nested loops:

```text
for i = 0 to N-1
    for j = 0 to N-1
        for k = 0 to N-1
            C[i][j] += A[i][k] × B[k][j]
```

Therefore, its computational complexity is:

```text
O(N³)
```

For:

```text
N = 4000
```

the number of iterations of the innermost computation is:

```text
4000³ = 64,000,000,000
```

This large computational workload makes matrix multiplication suitable for demonstrating the effects of parallel computing.

### 3.3 OpenMP Configuration

The OpenMP implementation uses:

* **Number of CPU threads:** 8
* The workload is divided among the eight threads by parallelizing the outer matrix loop.

Conceptually:

```text
4000 rows
    |
    +---- Thread 1
    +---- Thread 2
    +---- Thread 3
    +---- Thread 4
    +---- Thread 5
    +---- Thread 6
    +---- Thread 7
    +---- Thread 8
```

Each thread processes a portion of the matrix rows.

### 3.4 MPI Configuration

The MPI implementation uses:

* **Number of MPI processes:** 4

The workload is divided equally:

```text
4000 rows / 4 processes = 1000 rows per process
```

Therefore:

| MPI Rank   | Rows |
| ---------- | ---: |
| **Rank 0** | 1000 |
| **Rank 1** | 1000 |
| **Rank 2** | 1000 |
| **Rank 3** | 1000 |

The MPI setup consists of:

* Master
* Worker 1
* Worker 2
* Worker 3

The processes communicate using MPI message-passing operations.

### 3.5 CUDA Configuration

The CUDA implementation uses the following configuration:

| Parameter             |       Value |
| --------------------- | ----------: |
| **Matrix size**       | 4000 × 4000 |
| **Block dimensions**  |     16 × 16 |
| **Threads per block** |         256 |
| **Grid dimensions**   |   250 × 250 |
| **Total blocks**      |      62,500 |

The grid dimensions are calculated as:

```text
4000 / 16 = 250
```

Therefore:

```text
Grid = 250 × 250
```

and:

```text
250 × 250 = 62,500 blocks
```

Each block contains:

```text
16 × 16 = 256 threads
```

Each CUDA thread is responsible for computing one output element of the result matrix, subject to the boundary checks implemented in the kernel.


## 4. Build and Execution Protocols

This section describes how each implementation is compiled and executed. The four implementations use different compilers and execution environments because they use different parallel computing technologies.

---

### 4.1 Sequential Implementation

#### Explanation

The sequential implementation serves as the **baseline** for performance comparison. It uses a single CPU execution thread and does not require any parallel computing framework.

The program is compiled using GCC with the `-O2` optimization flag.

#### Step 1: Enter the WSL Environment

```bash
wsl
```

This starts the **Windows Subsystem for Linux (WSL)** environment, where the compilation and execution commands are performed.

#### Step 2: Verify GCC Installation

```bash
gcc --version
```

This checks whether the GCC compiler is installed and displays its version.

GCC is required to compile the C source code into an executable program.

#### Step 3: Create the Project Directory

```bash
mkdir -p ~/parallel_lab/sequential
cd ~/parallel_lab/sequential
```

The first command creates the `sequential` directory inside `parallel_lab`.

The `-p` option allows the command to create the parent directory if it does not already exist.

The second command moves into the newly created directory.

The expected structure is:

```text
parallel_lab/
└── sequential/
```

#### Step 4: Compile the Program

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
```

This command compiles `matrix_sequential.c`.

* `gcc` → invokes the GCC compiler
* `-O2` → enables compiler optimization
* `matrix_sequential.c` → input source file
* `-o matrix_sequential` → names the generated executable `matrix_sequential`

After successful compilation:

```text
matrix_sequential.c
        |
        v
      GCC
        |
        v
matrix_sequential
```

#### Step 5: Run the Benchmark

```bash
./matrix_sequential
```

The `./` indicates that the executable is located in the current directory.

The program performs the complete matrix multiplication and reports its execution time and correctness result.

This execution time is used as the **baseline** for calculating the speedup of OpenMP, MPI, and CUDA.

---

### 4.2 OpenMP Implementation

#### Explanation

The OpenMP implementation uses multiple CPU threads to execute parts of the matrix multiplication concurrently.

The program is compiled with the `-fopenmp` flag. This enables GCC to process OpenMP directives such as:

```c
#pragma omp parallel for
```

and link the program with the OpenMP runtime library.

#### Step 1: Check Available CPU Threads

```bash
nproc
```

This displays the number of processing units available to the Linux environment.

For this experiment, the implementation is configured to use **8 threads**.

#### Step 2: Configure the Number of OpenMP Threads

```bash
export OMP_NUM_THREADS=8
```

This sets the `OMP_NUM_THREADS` environment variable to `8`.

It tells the OpenMP runtime to create a team of eight threads when executing the parallel region.

Conceptually:

```text
Matrix rows
     |
     +---- Thread 1
     +---- Thread 2
     +---- Thread 3
     +---- Thread 4
     +---- Thread 5
     +---- Thread 6
     +---- Thread 7
     +---- Thread 8
```

Each thread works on a portion of the matrix rows.

#### Step 3: Compile the OpenMP Program

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

The important difference from the sequential compilation is:

```text
-fopenmp
```

This enables OpenMP support.

The command performs the following:

* `gcc` → invokes GCC
* `-O2` → enables compiler optimization
* `-fopenmp` → enables OpenMP
* `matrix_openmp.c` → OpenMP source file
* `-o matrix_openmp` → creates the executable

#### Step 4: Run the OpenMP Benchmark

```bash
./matrix_openmp
```

The program creates the configured OpenMP threads and divides the matrix multiplication workload among them.

The execution time is recorded and later compared with the sequential baseline.

---

### 4.3 MPI Implementation

#### Explanation

The MPI implementation uses **multiple processes** instead of shared-memory threads.

For this experiment, four MPI processes are used:

```text
Rank 0 → 1000 rows
Rank 1 → 1000 rows
Rank 2 → 1000 rows
Rank 3 → 1000 rows
```

The MPI setup consists of a master and three worker nodes.

Unlike OpenMP, MPI processes have separate memory spaces and communicate using MPI message-passing operations.

#### Step 1: Verify Worker Connectivity

```bash
ping -c 4 worker1
```

This checks whether the master system can communicate with `worker1`.

The same connectivity should be available for the other worker nodes.

The MPI experiment requires communication between the systems because processes may execute on different machines.

#### Step 2: Install OpenMPI

```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y
```

The first command updates the package information.

The second installs the OpenMPI runtime and development files.

The MPI development package provides the tools required to compile MPI programs, including `mpicc`.

#### Step 3: Configure Passwordless SSH

MPI uses SSH to start processes on the worker nodes.

Generate an SSH key:

```bash
ssh-keygen -t rsa
```

Then copy the public key to each worker:

```bash
ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3
```

This allows the master to connect to the worker nodes without repeatedly asking for an SSH password.

Test the connection, for example:

```bash
ssh worker1
```

If the connection succeeds without requesting a password, the SSH configuration is ready.

#### Step 4: Create the MPI Host File

Create a file named:

```text
hosts
```

with:

```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

This file tells `mpirun` which machines are available for the experiment and how many process slots are assigned to each machine.

The resulting configuration is:

```text
              Master
                 |
       +---------+---------+
       |         |         |
    Worker 1  Worker 2  Worker 3
```

#### Step 5: Compile the MPI Program

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

`mpicc` is the MPI compiler wrapper.

It automatically supplies the required MPI headers and libraries during compilation.

The resulting executable is:

```text
matrix_mpi
```

#### Step 6: Copy the Executable to Worker Nodes

```bash
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi
```

`scp` securely copies the compiled executable from the master to each worker.

This ensures that the executable exists at the expected path on every worker node.

#### Step 7: Run the MPI Benchmark

```bash
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

This starts the MPI experiment.

The important options are:

* `mpirun` → launches the MPI processes
* `-np 4` → starts four MPI processes
* `--hostfile hosts` → uses the machines listed in the host file
* `$HOME/matrix_mpi` → specifies the executable to run

The four processes divide the matrix workload and communicate using MPI operations such as:

```text
MPI_Bcast
    ↓
MPI_Scatter
    ↓
Local Matrix Multiplication
    ↓
MPI_Gather
```

The execution time is recorded for comparison with the sequential and OpenMP implementations.

---

### 4.4 CUDA Implementation

#### Explanation

The CUDA implementation uses an NVIDIA GPU to perform the matrix multiplication.

CUDA programs are compiled using NVIDIA's `nvcc` compiler.

The experiment uses:

```text
Block size = 16 × 16
Threads per block = 256
Grid size = 250 × 250
```

#### Step 1: Verify GPU Availability

```bash
nvidia-smi
```

This displays information about the NVIDIA GPU, including:

* GPU model
* Driver version
* GPU memory
* Current GPU utilization

A working NVIDIA GPU and driver installation are required for CUDA execution.

#### Step 2: Verify the CUDA Compiler

```bash
nvcc --version
```

This checks whether the NVIDIA CUDA compiler is installed and displays its version.

`nvcc` is required to compile `.cu` CUDA source files.

#### Step 3: Compile the CUDA Program

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

The command performs the following:

* `nvcc` → NVIDIA CUDA compiler
* `-O2` → compiler optimization
* `matrix_cuda.cu` → CUDA source file
* `-o matrix_cuda` → generated executable

The source contains both host-side CPU code and device-side GPU kernel code.

Conceptually:

```text
matrix_cuda.cu
       |
       v
     nvcc
       |
       +------ CPU Host Code
       |
       +------ GPU CUDA Kernel
       |
       v
  matrix_cuda
```

#### Step 4: Run the CUDA Benchmark

```bash
./matrix_cuda
```

The program performs the following general sequence:

```text
CPU Host Memory
      |
      | cudaMemcpyHostToDevice
      v
GPU Device Memory
      |
      | CUDA Kernel Launch
      v
GPU Threads Perform
Matrix Multiplication
      |
      | cudaMemcpyDeviceToHost
      v
CPU Host Memory
```

The program then reports the execution time and verifies the resulting matrix.

The CUDA execution time is compared with the sequential, OpenMP, and MPI execution times.

---

### 4.5 Overall Execution Flow

The four implementations should be executed independently using the same matrix size and input values.

```text
                    4000 × 4000 Matrix
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        Sequential      OpenMP          MPI
        1 CPU Core     8 Threads     4 Processes
             |             |             |
             +-------------+-------------+
                           |
                           v
                       CUDA GPU
                    16 × 16 Blocks
                           |
                           v
                 Execution Time
                 + Correctness
                           |
                           v
                 Performance Comparison
```

For every implementation, record:

1. **Compilation status**
2. **Execution time**
3. **Correctness / output value**
4. **Any execution errors**

The recorded execution times are then used to calculate the speedup relative to the sequential implementation.

## 5. Experimental Results

### Execution Time Measurements

| Framework      | Processing Topology             | Measured Execution Time (seconds) |
| -------------- | ------------------------------- | --------------------------------- |
| **Sequential** | Single CPU Core Execution       | **606.987331**                    |
| **OpenMP**     | 8 CPU Multi-Threaded Cores      | **40.574496**                     |
| **MPI**        | 4 Distributed Cluster Processes | **223.691390**                    |
| **CUDA**       | Massively Parallel GPU Threads  | **0.225428**                      |

*Note on CUDA Timings:*

- **CUDA Kernel Only Duration:** 0.188994 seconds
- **CUDA Total Phase Duration (including transfers):** 0.225428 seconds

### Correctness Verification

#### Explanation

Execution accuracy is verified by asserting that the computed output matches the expected mathematical target ().

| Implementation | Theoretical Target | Reported Value | Verification Outcome            |
| -------------- | ------------------ | -------------- | ------------------------------- |
| **Sequential** | 4000.00            | 4000.00        | **Verified / Passed**           |
| **OpenMP**     | 4000.00            | 4000.00        | **Verified / Passed**           |
| **MPI**        | 4000.00            | 4000.00        | **Verified / Passed**           |
| **CUDA**       | 4000.00            | 0.00           | **Failed / Requires Debugging** |

### Speedup Metrics

Speedup is calculated relative to the single-core sequential baseline:

| Implementation | Parallel Compute Units | Runtime (s) | Calculated Speedup | Parallel Efficiency |
| -------------- | ---------------------- | ----------- | ------------------ | ------------------- |
| **Sequential** | 1 CPU Core             | 606.987331  | 1.00×              | 100%                |
| **OpenMP**     | 8 CPU Threads          | 40.574496   | 14.96×             | 187%                |
| **MPI**        | 4 Processes            | 223.691390  | 2.71×              | 68%                 |
| **CUDA**       | GPU Threads            | 0.225428    | *(2692.60×)*       | *Unvalidated*       |

## 6. Detailed Technical Analysis & Overhead Explanation

### Sequential Baseline Mechanics

#### Explanation

The sequential program required over 10 minutes (606.98s) to finish  arithmetic operations. Beyond raw compute load, the execution is slowed down by poor memory cache usage.

The inner loop accesses  along column bounds. In C, matrices are stored in memory row-by-row (row-major order). Jumping through columns causes the CPU to miss its fast L1/L2 memory caches frequently, forcing it to retrieve data directly from slower system RAM.

### OpenMP Mechanics & Cache Interactions

#### Explanation

OpenMP reduced runtime down to 40.57 seconds, delivering a **14.96× speedup** across 8 threads.

#### Superlinear Speedup Phenomenon

The speedup ratio exceeds the physical thread count (). This occurs due to memory cache effects:

1. When matrix rows are partitioned across 8 separate CPU cores, each core handles a smaller chunk of data.
2. These smaller data chunks fit entirely within the fast L1/L2/L3 caches built into each core.
3. This reduces main RAM memory access stalls compared to the single-threaded run, resulting in a speedup higher than the raw core count alone would suggest.

### MPI Mechanics & Network Bottlenecks

#### Explanation

MPI recorded a runtime of 223.69 seconds (**2.71× speedup** across 4 processes, representing a \~68% parallel efficiency).

#### Network & Communication Overheads

Unlike OpenMP where all threads access shared RAM directly, MPI processes do not share memory:

1. **Data Transfer Costs:** Matrix  () must be broadcast over the network to all worker nodes via `MPI_Bcast`.
2. **Scatter/Gather Delays:** `MPI_Scatter` splits Matrix  across processes, and `MPI_Gather` collects partial results back to Rank 0.
3. **Synchronization Overhead:** Slower network packets force faster compute processes to sit idle at collective boundaries until transfer phases complete.

### CUDA Mechanics & Diagnostic Failure Analysis

#### Explanation

While CUDA recorded a total phase duration of **0.225428 seconds**, the verification check returned **0.00** instead of **4000.00**.

#### Failure Diagnostics & Unvalidated Speedup

A value of 0.00 indicates a kernel or memory management failure:

- **Root Causes:** Common causes include out-of-bounds array access, incorrect grid-block parameter bounds, unallocated VRAM pointers, or failing to call `cudaMemcpy` from Device back to Host before checking array values.
- **Performance Impact:** An unvalidated timing speedup (such as ) cannot be treated as a valid performance metric. Failed GPU kernels often exit immediately without completing the underlying arithmetic, leading to artificially fast runtimes.

## 7. Conclusion

This project evaluated matrix multiplication using four distinct computing approaches: Sequential CPU, OpenMP multi-threading, MPI distributed message passing, and CUDA GPU acceleration.

### Summary Findings

1. **Shared Memory Efficiency:** Among the verified implementations, OpenMP delivered the highest real-world performance gain (**14.96× speedup** on 8 threads) due to shared-memory pointer access and improved CPU cache hit rates.
2. **Network Bottleneck Impact:** MPI achieved a lower speedup (**2.71× speedup** on 4 processes) because sending  data payloads over network channels introduces significant communication latency.
3. **Correctness Verification Priority:** Benchmark timings must always be verified for correctness. The CUDA implementation ran in 0.22 seconds, but its failure to pass output validation demonstrates that speedup metrics are meaningless without accurate arithmetic results.

## 8. Repository Structure

```
Parallel-Matrix-Multiplication/
├── README.md
├── .gitignore
│
├── sequential/
│   └── matrix_sequential.c
│
├── openmp/
│   └── matrix_openmp.c
│
├── mpi/
│   ├── matrix_mpi.c
│   └── hosts
│
├── cuda/
│   └── matrix_cuda.cu
│
└── results/
    ├── execution_time.png
    └── speedup.png

```
