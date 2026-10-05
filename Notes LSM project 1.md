MPI rank = one independent process

**parallelisation** = give different ranks different work
**communication** = move results between those processes
**load imbalance** = some ranks finish much later than others
**static scheduling** = decide everyone's work beforehand
**dynamic scheduling** = give workers new work as they finish
**blocking communication** = process waits for communication
**non-blocking communication** = communication can be started and completed later
**scaling** = how runtime changes as we add more processes
**overhead** = communication/setup costs that stop speedup being perfect


**03-10 | where are we right now?**

Serial Mandelbrot
      ↓
Basic MPI splitting
      ↓
Blocking communication
      ↓
Chunk-based static load balancing
      ↓
Non-blocking communication
      ↓
YOU ARE HERE

**next steps**
- Get one **clean blocking version** working.
- Get one **clean non-blocking version** working.
	- created one script in which the elements of the code are called as separate functions. this way we can run all timings from the same script to ensure e.g. the mandelbrot computation is identical for all versions, but then simply the blocking communication and non-blocking is the only part that changes.
	- furthermore, the script can be called in the hpc using a simple change of arguments in the command line, making it simpler to call.

- Time the full calculation consistently.
	- timing blocking/nonblocking with 1-16 nodes:

parameters for tests:
``chunk_size = 10
``size       = 1000x1000
``xlim       = -2.2:0.75
``ylim       = -1.3:1.3

| VERSION      | n = 1       | n = 2       | n = 4      | n = 8      | n = 16     |
| ------------ | ----------- | ----------- | ---------- | ---------- | ---------- |
| blocking1    | 33.5633     | 14.4337     | 8.5551     | 4.3340     | 2.6872     |
| blocking2    | 33.6869     | 15.7995     | 8.9795     | 4.3356     | 2.5705     |
| blocking3    | 34.5248     | 15.0677     | 8.7791     | 4.3898     | 2.4746     |
| **blocking** | **33.9250** | **15.1003** | **8.7712** | **4.3531** | **2.5774** |
| nonblock1    | 34.0196     | 18.4768     | 8.4189     | 4.3238     | 2.4087     |
| nonblock2    | 34.9135     | 15.3848     | 8.4598     | 4.1869     | 2.8292     |
| nonblock3    | 31.2614     | 15.9636     | 8.7134     | 4.3006     | 2.3551     |
| **nonblock** | **33.3982** | **16.6084** | **8.5307** | **4.2704** | **2.5310** |
    
- Compare different `chunk_size`s, e.g.
    
    ```
    1, 5, 10, 25, 50, 100
    ```
    
- See whether chunking improves load balance.
- Compare blocking vs non-blocking.
- Move the jobs to DTU HPC and test one node vs multiple nodes.
