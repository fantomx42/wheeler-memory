/*
 * Wheeler Memory — HIP GPU Kernel for Cellular Automata Evolution
 *
 * Exactly replicates dynamics.py + oscillation.py on AMD GPUs via ROCm/HIP.
 * One block per frame, 256 threads, 16 cells/thread (4096 / 256 = 16).
 *
 * Shared memory per block (~37 KB, fits in 64 KB RDNA4 LDS):
 *   sframe[4096]  float   16 KB  – current frame
 *   snext[4096]   float   16 KB  – next frame
 *   sdelta[256]   float    1 KB  – reduction scratch
 *   sroles[4096]  int8_t   4 KB  – roles / change-flags (reused)
 *   sdone / sfinal_state / sfinal_ticks  (3 ints, 12 B)
 *
 * Global memory role-history ring:  batch_size × 20 × 4096  int8_t
 *   Roles are stored from the OUTPUT frame (snext) to match
 *   Python's get_cell_roles(history[i]) called on post-update frames.
 *   Storage starts at iter = 41 in slot (iter-41) % 20.
 *   At oscillation check (iter ≥ 60, every 10): oldest_slot = (iter-60) % 20.
 *
 * Build:
 *   hipcc --offload-arch=gfx1201 -O3 -shared -fPIC -o libwheeler_ca.so ca_kernel.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define GRID_SIZE          4096   /* 64 × 64               */
#define GRID_W             64
#define THREADS_PER_BLOCK  256
#define CELLS_PER_THREAD   16    /* GRID_SIZE / THREADS_PER_BLOCK */
#define OSC_WINDOW         20    /* ring-buffer depth             */
#define OSC_START          41    /* first iter whose roles we keep */
#define STABILITY_THRESHOLD 1e-4f
#define OSC_MIN_CELLS      41    /* 1 % of 4096, rounded up       */

/* ─── Main evolution kernel ────────────────────────────────────────────────── */

__global__ void ca_evolve_kernel(
    const float* __restrict__ frames_in,   /* B × GRID_SIZE  float32 */
    float*       __restrict__ frames_out,  /* B × GRID_SIZE  float32 */
    int*         __restrict__ ticks_out,   /* B              int32   */
    int*         __restrict__ states_out,  /* B              int32   */
    int8_t*      __restrict__ role_hist,   /* B×20×GRID_SIZE int8_t  */
    int max_iters)
{
    /* ── Shared memory ─────────────────────────────────────────────────────── */
    __shared__ float  sframe[GRID_SIZE];
    __shared__ float  snext [GRID_SIZE];
    __shared__ float  sdelta[THREADS_PER_BLOCK];
    __shared__ int8_t sroles[GRID_SIZE];
    __shared__ int    sdone;
    __shared__ int    sfinal_state;
    __shared__ int    sfinal_ticks;

    int b         = (int)blockIdx.x;
    int tid       = (int)threadIdx.x;
    int base_cell = tid * CELLS_PER_THREAD;   /* first cell for this thread */

    /* Init control variables */
    if (tid == 0) {
        sdone        = 0;
        sfinal_state = 2;          /* CHAOTIC default */
        sfinal_ticks = max_iters;
    }

    /* Load initial frame into shared memory */
    for (int j = 0; j < CELLS_PER_THREAD; j++)
        sframe[base_cell + j] = frames_in[(size_t)b * GRID_SIZE + base_cell + j];

    __syncthreads();

    /* ── Main iteration loop ─────────────────────────────────────────────── */
    for (int iter = 0; iter < max_iters; iter++) {

        /* ── Phase 1: CA update ──────────────────────────────────────────── *
         * Each thread writes 16 cells of snext and accumulates local_delta.  *
         * Roles (from sframe) are NOT stored here; see Phase 2b.             */
        float local_delta = 0.0f;

        for (int j = 0; j < CELLS_PER_THREAD; j++) {
            int k   = base_cell + j;
            int row = k / GRID_W;
            int col = k % GRID_W;

            int up_r    = (row == 0)          ? GRID_W - 1 : row - 1;
            int dn_r    = (row == GRID_W - 1) ? 0          : row + 1;
            int lf_c    = (col == 0)          ? GRID_W - 1 : col - 1;
            int rt_c    = (col == GRID_W - 1) ? 0          : col + 1;

            float v     = sframe[k];
            float n_up  = sframe[up_r * GRID_W + col];
            float n_dn  = sframe[dn_r * GRID_W + col];
            float n_lf  = sframe[row  * GRID_W + lf_c];
            float n_rt  = sframe[row  * GRID_W + rt_c];

            bool is_max = (v >= n_up) & (v >= n_dn) & (v >= n_lf) & (v >= n_rt);
            bool is_min = (v <= n_up) & (v <= n_dn) & (v <= n_lf) & (v <= n_rt);

            float delta;
            if (is_max) {
                delta = (1.0f - v) * 0.35f;
            } else if (is_min) {
                delta = (-1.0f - v) * 0.35f;
            } else {
                float max_n = fmaxf(fmaxf(n_up, n_dn), fmaxf(n_lf, n_rt));
                delta = (max_n - v) * 0.20f;
            }

            float new_v   = fminf(fmaxf(v + delta, -1.0f), 1.0f);
            snext[k]      = new_v;
            local_delta  += fabsf(new_v - v);
        }

        /* ── Phase 2: Delta reduction ────────────────────────────────────── *
         * __syncthreads() here also ensures all snext values are visible.    */
        sdelta[tid] = local_delta;
        __syncthreads();

        for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) sdelta[tid] += sdelta[tid + s];
            __syncthreads();
        }
        /* sdelta[0] = sum of |delta| over all GRID_SIZE cells */

        /* ── Phase 2b: Role computation from snext ───────────────────────── *
         * Must use snext (post-update frame) to match Python's               *
         * get_cell_roles(history[i]) which is called on output frames.       *
         * snext is fully written + all-threads-visible after Phase 2 syncs.  */
        for (int j = 0; j < CELLS_PER_THREAD; j++) {
            int k   = base_cell + j;
            int row = k / GRID_W;
            int col = k % GRID_W;

            int up_r = (row == 0)          ? GRID_W - 1 : row - 1;
            int dn_r = (row == GRID_W - 1) ? 0          : row + 1;
            int lf_c = (col == 0)          ? GRID_W - 1 : col - 1;
            int rt_c = (col == GRID_W - 1) ? 0          : col + 1;

            float v    = snext[k];
            float n_up = snext[up_r * GRID_W + col];
            float n_dn = snext[dn_r * GRID_W + col];
            float n_lf = snext[row  * GRID_W + lf_c];
            float n_rt = snext[row  * GRID_W + rt_c];

            bool is_max = (v >= n_up) & (v >= n_dn) & (v >= n_lf) & (v >= n_rt);
            bool is_min = (v <= n_up) & (v <= n_dn) & (v <= n_lf) & (v <= n_rt);

            sroles[k] = is_max ? (int8_t)1 : (is_min ? (int8_t)-1 : (int8_t)0);
        }
        /* No sync needed: ring write reads only this thread's own cells */

        /* ── Phase 3: Ring write ─────────────────────────────────────────── */
        if (iter >= OSC_START) {
            int slot = (iter - OSC_START) % OSC_WINDOW;
            int8_t* rptr = role_hist
                         + (size_t)b    * OSC_WINDOW * GRID_SIZE
                         + (size_t)slot * GRID_SIZE;
            for (int j = 0; j < CELLS_PER_THREAD; j++)
                rptr[base_cell + j] = sroles[base_cell + j];
        }

        /* ── Phase 4: Swap sframe ← snext ───────────────────────────────── */
        for (int j = 0; j < CELLS_PER_THREAD; j++)
            sframe[base_cell + j] = snext[base_cell + j];
        __syncthreads();   /* new sframe + global ring writes visible to all */

        /* ── Phase 5: Convergence check ──────────────────────────────────── */
        if (tid == 0) {
            float mean_delta = sdelta[0] / (float)GRID_SIZE;
            if (mean_delta < STABILITY_THRESHOLD) {
                sdone        = 1;
                sfinal_state = 0;       /* CONVERGED */
                sfinal_ticks = iter + 1;
            }
        }
        __syncthreads();
        if (sdone) break;

        /* ── Phase 6: Oscillation check (iter > 50, every 10 iters) ──────── *
         * First possible check: iter = 60 (>50 and %10==0).                  *
         * Ring holds iter=41..current (20 frames).                           */
        if (iter > 50 && (iter % 10) == 0) {
            int oldest_slot = (iter - 60) % OSC_WINDOW;  /* iter>=60, always ≥ 0 */
            size_t bbase    = (size_t)b * OSC_WINDOW * GRID_SIZE;

            /* Phase A: compute change flags into sroles.
             * sroles[k] = 1 if any t in 1..19 has role[t][k] != role[oldest][k],
             * i.e. the cell changed role at some point in the 20-frame window. */
            for (int j = 0; j < CELLS_PER_THREAD; j++) {
                int k           = base_cell + j;
                int8_t oldest_r = role_hist[bbase + (size_t)oldest_slot * GRID_SIZE + k];
                int8_t changed  = 0;
                for (int t = 1; t < OSC_WINDOW; t++) {
                    int slot_t = (oldest_slot + t) % OSC_WINDOW;
                    if (role_hist[bbase + (size_t)slot_t * GRID_SIZE + k] != oldest_r) {
                        changed = 1;
                        break;
                    }
                }
                sroles[k] = changed;
            }
            __syncthreads();  /* all change flags written before Phase B reads sroles */

            /* Phase B: for each period p in 2..10, count oscillating cells.
             * oscillating[k] = sroles[k] && (roles[t] == roles[t+p] for all t). */
            for (int p = 2; p <= 10; p++) {
                int n_checks  = OSC_WINDOW - p;   /* = 20 - p */
                int local_cnt = 0;

                for (int j = 0; j < CELLS_PER_THREAD; j++) {
                    int k = base_cell + j;
                    if (!sroles[k]) continue;     /* constant cell, skip */

                    bool matches = true;
                    for (int t = 0; t < n_checks && matches; t++) {
                        int slot_t  = (oldest_slot + t)     % OSC_WINDOW;
                        int slot_tp = (oldest_slot + t + p) % OSC_WINDOW;
                        int8_t rt   = role_hist[bbase + (size_t)slot_t  * GRID_SIZE + k];
                        int8_t rtp  = role_hist[bbase + (size_t)slot_tp * GRID_SIZE + k];
                        if (rt != rtp) matches = false;
                    }
                    if (matches) local_cnt++;
                }

                /* Reduce local_cnt across block (reuse sdelta) */
                sdelta[tid] = (float)local_cnt;
                __syncthreads();
                for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
                    if (tid < s) sdelta[tid] += sdelta[tid + s];
                    __syncthreads();
                }

                if (tid == 0) {
                    if ((int)sdelta[0] >= OSC_MIN_CELLS) {
                        sdone        = 1;
                        sfinal_state = 1;       /* OSCILLATING */
                        sfinal_ticks = iter + 1;
                    }
                }
                __syncthreads();
                if (sdone) break;
            }
        }

        if (sdone) break;
    } /* main loop */

    /* Write final frame and metadata.
     * sframe always holds the latest frame because the swap (Phase 4) happens
     * before the convergence/oscillation break. */
    for (int j = 0; j < CELLS_PER_THREAD; j++)
        frames_out[(size_t)b * GRID_SIZE + base_cell + j] = sframe[base_cell + j];

    if (tid == 0) {
        ticks_out [b] = sfinal_ticks;
        states_out[b] = sfinal_state;
    }
}


/* ─── Host C API (extern "C" for ctypes) ────────────────────────────────────── */

extern "C" {

/**
 * ca_evolve_batch — evolve B frames in parallel on GPU.
 *
 * @param frames_in   B × 4096  float32, row-major, values in [-1, 1]
 * @param frames_out  B × 4096  float32, converged/oscillated attractors
 * @param ticks_out   B         int32,   iterations until termination
 * @param states_out  B         int32,   0=CONVERGED 1=OSCILLATING 2=CHAOTIC
 * @param batch_size  Number of frames (= B)
 * @param max_iters   Max iterations before declaring CHAOTIC
 * @return 0 on success, -1 on HIP error
 */
int ca_evolve_batch(const float* frames_in,
                    float*       frames_out,
                    int*         ticks_out,
                    int*         states_out,
                    int          batch_size,
                    int          max_iters)
{
    if (batch_size <= 0) return 0;

    size_t frame_bytes = (size_t)batch_size * GRID_SIZE  * sizeof(float);
    size_t role_bytes  = (size_t)batch_size * OSC_WINDOW * GRID_SIZE * sizeof(int8_t);
    size_t meta_bytes  = (size_t)batch_size * sizeof(int);

    float*  d_in     = nullptr;
    float*  d_out    = nullptr;
    int*    d_ticks  = nullptr;
    int*    d_states = nullptr;
    int8_t* d_roles  = nullptr;
    int     retval   = 0;
    hipError_t err;

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        err = (call);                                                        \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d — %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(err));             \
            retval = -1; goto cleanup;                                       \
        }                                                                    \
    } while (0)

    HIP_CHECK(hipMalloc(&d_in,     frame_bytes));
    HIP_CHECK(hipMalloc(&d_out,    frame_bytes));
    HIP_CHECK(hipMalloc(&d_ticks,  meta_bytes));
    HIP_CHECK(hipMalloc(&d_states, meta_bytes));
    HIP_CHECK(hipMalloc(&d_roles,  role_bytes));

    HIP_CHECK(hipMemcpy(d_in, frames_in, frame_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_roles, 0, role_bytes));

    hipLaunchKernelGGL(ca_evolve_kernel,
                       dim3(batch_size), dim3(THREADS_PER_BLOCK), 0, 0,
                       d_in, d_out, d_ticks, d_states, d_roles, max_iters);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(frames_out, d_out,    frame_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ticks_out,  d_ticks,  meta_bytes,  hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(states_out, d_states, meta_bytes,  hipMemcpyDeviceToHost));

cleanup:
    if (d_in)     (void)hipFree(d_in);
    if (d_out)    (void)hipFree(d_out);
    if (d_ticks)  (void)hipFree(d_ticks);
    if (d_states) (void)hipFree(d_states);
    if (d_roles)  (void)hipFree(d_roles);

    return retval;
}


/**
 * ca_evolve_single — convenience wrapper for a single frame.
 */
int ca_evolve_single(const float* frame_in,
                     float*       frame_out,
                     int*         ticks_out,
                     int*         state_out,
                     int          max_iters)
{
    return ca_evolve_batch(frame_in, frame_out, ticks_out, state_out, 1, max_iters);
}

}  /* extern "C" */
