/*
 * Wheeler Memory — HIP GPU Kernel v2: Variable Grid Size + Global Memory
 *
 * Supports grids up to 1000x1000 (1M cells) using global memory instead of
 * shared-memory-only approach of v1 (which was limited to 64x64).
 *
 * Architecture: 5 separate kernels launched in a host-side iteration loop.
 * One thread per cell with 2D spatial tiling (16x16 blocks) for L2 locality.
 *
 * Kernels:
 *   1. ca_step_kernel         — CA update + atomic delta accumulation
 *   2. convergence_check_kernel — mean_delta check, zero accumulators
 *   3. role_compute_kernel     — classify roles, write to ring buffer
 *   4. oscillation_detect_kernel — period 2-10 check with block reduction
 *   5. oscillation_finalize_kernel — check osc_counts, update status
 *
 * Global memory buffers:
 *   d_grid_A / d_grid_B (ping-pong)  float32 [B x W x W]
 *   d_role_ring                       int8    [B x 20 x W x W]
 *   d_delta_sums                      float32 [B]
 *   d_status / d_ticks                int32   [B]
 *   d_osc_counts                      int32   [B x 9]
 *
 * Build:
 *   hipcc --offload-arch=gfx1201 -O3 -shared -fPIC -o libwheeler_ca_v2.so ca_kernel_v2.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define OSC_WINDOW  20    /* ring-buffer depth             */
#define OSC_START   41    /* first iter whose roles we keep */
#define OSC_PERIODS  9    /* periods 2..10                  */

/* Status codes */
#define STATUS_RUNNING     -1
#define STATUS_CONVERGED    0
#define STATUS_OSCILLATING  1
#define STATUS_CHAOTIC      2


/* ─── Kernel 1: CA step + atomic delta accumulation ───────────────────────── */

__global__ void ca_step_kernel(
    const float* __restrict__ grid_src,    /* [B x W x W] */
    float*       __restrict__ grid_dst,    /* [B x W x W] */
    float*       __restrict__ delta_sums,  /* [B]         */
    const int*   __restrict__ status,      /* [B]         */
    int grid_w,
    int grid_cells)  /* grid_w * grid_w */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (col >= grid_w || row >= grid_w) return;

    size_t gc = (size_t)grid_cells;
    size_t k  = (size_t)b * gc + (size_t)row * grid_w + col;

    /* Terminated frames: copy src→dst to keep both buffers in sync */
    if (status[b] != STATUS_RUNNING) {
        grid_dst[k] = grid_src[k];
        return;
    }

    int up_r = (row == 0)          ? grid_w - 1 : row - 1;
    int dn_r = (row == grid_w - 1) ? 0          : row + 1;
    int lf_c = (col == 0)          ? grid_w - 1 : col - 1;
    int rt_c = (col == grid_w - 1) ? 0          : col + 1;

    float v    = grid_src[k];
    float n_up = grid_src[(size_t)b * gc + (size_t)up_r * grid_w + col];
    float n_dn = grid_src[(size_t)b * gc + (size_t)dn_r * grid_w + col];
    float n_lf = grid_src[(size_t)b * gc + (size_t)row  * grid_w + lf_c];
    float n_rt = grid_src[(size_t)b * gc + (size_t)row  * grid_w + rt_c];

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

    float new_v = fminf(fmaxf(v + delta, -1.0f), 1.0f);
    grid_dst[k] = new_v;

    atomicAdd(&delta_sums[b], fabsf(new_v - v));
}


/* ─── Kernel 2: Convergence check + zero accumulator ─────────────────────── */

__global__ void convergence_check_kernel(
    float*       __restrict__ delta_sums,  /* [B]  */
    int*         __restrict__ status,      /* [B]  */
    int*         __restrict__ ticks,       /* [B]  */
    int grid_cells,
    int current_iter,
    float stability_threshold)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    /* Note: caller must ensure b < batch_size via grid launch config */

    if (status[b] != STATUS_RUNNING) {
        delta_sums[b] = 0.0f;
        return;
    }

    float mean_delta = delta_sums[b] / (float)grid_cells;
    if (mean_delta < stability_threshold) {
        status[b] = STATUS_CONVERGED;
        ticks[b]  = current_iter + 1;
    }

    delta_sums[b] = 0.0f;
}


/* ─── Kernel 3: Role computation + ring buffer write ─────────────────────── */

__global__ void role_compute_kernel(
    const float* __restrict__ grid,        /* [B x W x W]  — post-CA-step frame */
    int8_t*      __restrict__ role_ring,   /* [B x 20 x W*W] */
    const int*   __restrict__ status,      /* [B] */
    int grid_w,
    int grid_cells,
    int ring_slot)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    if (col >= grid_w || row >= grid_w) return;
    if (status[b] != STATUS_RUNNING) return;

    size_t gc = (size_t)grid_cells;
    size_t k  = (size_t)b * gc + (size_t)row * grid_w + col;
    int cell_idx = row * grid_w + col;

    int up_r = (row == 0)          ? grid_w - 1 : row - 1;
    int dn_r = (row == grid_w - 1) ? 0          : row + 1;
    int lf_c = (col == 0)          ? grid_w - 1 : col - 1;
    int rt_c = (col == grid_w - 1) ? 0          : col + 1;

    float v    = grid[k];
    float n_up = grid[(size_t)b * gc + (size_t)up_r * grid_w + col];
    float n_dn = grid[(size_t)b * gc + (size_t)dn_r * grid_w + col];
    float n_lf = grid[(size_t)b * gc + (size_t)row  * grid_w + lf_c];
    float n_rt = grid[(size_t)b * gc + (size_t)row  * grid_w + rt_c];

    bool is_max = (v >= n_up) & (v >= n_dn) & (v >= n_lf) & (v >= n_rt);
    bool is_min = (v <= n_up) & (v <= n_dn) & (v <= n_lf) & (v <= n_rt);

    int8_t role = is_max ? (int8_t)1 : (is_min ? (int8_t)-1 : (int8_t)0);

    /* Write to ring: [b][ring_slot][cell_idx] */
    role_ring[(size_t)b * OSC_WINDOW * gc + (size_t)ring_slot * gc + cell_idx] = role;
}


/* ─── Kernel 4: Oscillation detection (block-level reduction) ────────────── */

__global__ void oscillation_detect_kernel(
    const int8_t* __restrict__ role_ring,   /* [B x 20 x W*W] */
    int*          __restrict__ osc_counts,  /* [B x 9]         */
    const int*    __restrict__ status,      /* [B]             */
    int grid_w,
    int grid_cells,
    int oldest_slot)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    /* Shared memory for block-level reduction: 9 periods x 256 threads max */
    extern __shared__ int s_counts[];  /* [OSC_PERIODS x blockDim.x * blockDim.y] */

    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    /* Initialize shared memory counts to 0 */
    for (int p = 0; p < OSC_PERIODS; p++)
        s_counts[p * block_threads + local_tid] = 0;

    bool valid = (col < grid_w && row < grid_w && status[b] == STATUS_RUNNING);

    if (valid) {
        size_t gc = (size_t)grid_cells;
        int cell_idx = row * grid_w + col;
        size_t bbase = (size_t)b * OSC_WINDOW * gc;

        /* Check if cell changed at all in the window */
        int8_t oldest_r = role_ring[bbase + (size_t)oldest_slot * gc + cell_idx];
        bool changed = false;
        for (int t = 1; t < OSC_WINDOW; t++) {
            int slot_t = (oldest_slot + t) % OSC_WINDOW;
            if (role_ring[bbase + (size_t)slot_t * gc + cell_idx] != oldest_r) {
                changed = true;
                break;
            }
        }

        if (changed) {
            /* Check each period p in 2..10 */
            for (int p = 2; p <= 10; p++) {
                int n_checks = OSC_WINDOW - p;
                bool matches = true;
                for (int t = 0; t < n_checks && matches; t++) {
                    int slot_t  = (oldest_slot + t)     % OSC_WINDOW;
                    int slot_tp = (oldest_slot + t + p) % OSC_WINDOW;
                    int8_t rt  = role_ring[bbase + (size_t)slot_t  * gc + cell_idx];
                    int8_t rtp = role_ring[bbase + (size_t)slot_tp * gc + cell_idx];
                    if (rt != rtp) matches = false;
                }
                if (matches)
                    s_counts[(p - 2) * block_threads + local_tid] = 1;
            }
        }
    }
    __syncthreads();

    /* Block-level reduction for each period */
    for (int p = 0; p < OSC_PERIODS; p++) {
        int offset = p * block_threads;
        for (int s = block_threads / 2; s > 0; s >>= 1) {
            if (local_tid < s)
                s_counts[offset + local_tid] += s_counts[offset + local_tid + s];
            __syncthreads();
        }
        if (local_tid == 0 && s_counts[offset] > 0) {
            atomicAdd(&osc_counts[b * OSC_PERIODS + p], s_counts[offset]);
        }
    }
}


/* ─── Kernel 5: Oscillation finalize ─────────────────────────────────────── */

__global__ void oscillation_finalize_kernel(
    int*       __restrict__ osc_counts,  /* [B x 9] */
    int*       __restrict__ status,      /* [B]     */
    int*       __restrict__ ticks,       /* [B]     */
    int grid_cells,
    int current_iter)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    /* Note: caller ensures b < batch_size */

    if (status[b] != STATUS_RUNNING) {
        /* Zero counts for terminated frames */
        for (int p = 0; p < OSC_PERIODS; p++)
            osc_counts[b * OSC_PERIODS + p] = 0;
        return;
    }

    /* 1% of grid cells, rounded up */
    int min_cells = (grid_cells + 99) / 100;

    for (int p = 0; p < OSC_PERIODS; p++) {
        if (osc_counts[b * OSC_PERIODS + p] >= min_cells) {
            status[b] = STATUS_OSCILLATING;
            ticks[b]  = current_iter + 1;
            break;
        }
    }

    /* Zero counts for next round */
    for (int p = 0; p < OSC_PERIODS; p++)
        osc_counts[b * OSC_PERIODS + p] = 0;
}


/* ─── Host C API ─────────────────────────────────────────────────────────── */

extern "C" {

/* HIP error check macro */
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t _err = (call);                                            \
        if (_err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d — %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(_err));             \
            retval = -1; goto cleanup;                                       \
        }                                                                    \
    } while (0)


/**
 * ca_query_vram — estimate VRAM needed for a given batch + grid config.
 */
size_t ca_query_vram(int batch_size, int grid_w)
{
    size_t gc = (size_t)grid_w * grid_w;
    size_t B  = (size_t)batch_size;

    size_t grid_bytes  = B * gc * sizeof(float);         /* one grid plane */
    size_t ring_bytes  = B * OSC_WINDOW * gc * sizeof(int8_t);
    size_t delta_bytes = B * sizeof(float);
    size_t meta_bytes  = B * sizeof(int);
    size_t osc_bytes   = B * OSC_PERIODS * sizeof(int);

    /* 2 grid planes (ping-pong) + ring + delta + status + ticks + osc_counts */
    return 2 * grid_bytes + ring_bytes + delta_bytes + 2 * meta_bytes + osc_bytes;
}


/**
 * ca_evolve_batch_v2 — evolve B frames on GPU with variable grid size.
 *
 * @param frames_in           B x grid_w*grid_w  float32 row-major
 * @param frames_out          B x grid_w*grid_w  float32 output attractors
 * @param ticks_out           B                  int32 iterations until termination
 * @param states_out          B                  int32 0=CONVERGED 1=OSCILLATING 2=CHAOTIC
 * @param batch_size          Number of frames
 * @param grid_w              Grid width (64, 128, 256, 512, 1000, etc.)
 * @param max_iters           Max iterations before CHAOTIC
 * @param stability_threshold Convergence threshold for mean |delta|
 * @return 0 on success, -1 on HIP error
 */
int ca_evolve_batch_v2(
    const float* frames_in,
    float*       frames_out,
    int*         ticks_out,
    int*         states_out,
    int          batch_size,
    int          grid_w,
    int          max_iters,
    float        stability_threshold)
{
    if (batch_size <= 0 || grid_w <= 0) return 0;

    int grid_cells = grid_w * grid_w;
    size_t gc      = (size_t)grid_cells;
    size_t B       = (size_t)batch_size;

    size_t grid_bytes  = B * gc * sizeof(float);
    size_t ring_bytes  = B * OSC_WINDOW * gc * sizeof(int8_t);
    size_t delta_bytes = B * sizeof(float);
    size_t meta_bytes  = B * sizeof(int);
    size_t osc_bytes   = B * OSC_PERIODS * sizeof(int);

    float*  d_grid_A    = nullptr;  /* ping */
    float*  d_grid_B    = nullptr;  /* pong */
    int8_t* d_role_ring = nullptr;
    float*  d_delta     = nullptr;
    int*    d_status    = nullptr;
    int*    d_ticks     = nullptr;
    int*    d_osc       = nullptr;
    int     retval      = 0;

    /* Declare all variables before first goto-capable HIP_CHECK */
    dim3 block_2d(16, 16);   /* 256 threads per block */
    dim3 grid_2d((grid_w + 15) / 16, (grid_w + 15) / 16, batch_size);
    size_t osc_smem = OSC_PERIODS * 256 * sizeof(int);
    int conv_threads = (batch_size < 256) ? batch_size : 256;
    int conv_blocks  = (batch_size + conv_threads - 1) / conv_threads;
    float* src = nullptr;
    float* dst = nullptr;
    int* h_poll_status = (int*)malloc(meta_bytes);
    int* h_init_status = (int*)malloc(meta_bytes);
    int* h_init_ticks  = (int*)malloc(meta_bytes);
    bool all_done = false;

    HIP_CHECK(hipMalloc(&d_grid_A,    grid_bytes));
    HIP_CHECK(hipMalloc(&d_grid_B,    grid_bytes));
    HIP_CHECK(hipMalloc(&d_role_ring, ring_bytes));
    HIP_CHECK(hipMalloc(&d_delta,     delta_bytes));
    HIP_CHECK(hipMalloc(&d_status,    meta_bytes));
    HIP_CHECK(hipMalloc(&d_ticks,     meta_bytes));
    HIP_CHECK(hipMalloc(&d_osc,       osc_bytes));

    /* Copy input frames to grid A */
    HIP_CHECK(hipMemcpy(d_grid_A, frames_in, grid_bytes, hipMemcpyHostToDevice));

    /* Initialize buffers */
    HIP_CHECK(hipMemset(d_role_ring, 0, ring_bytes));
    HIP_CHECK(hipMemset(d_delta,     0, delta_bytes));
    HIP_CHECK(hipMemset(d_osc,       0, osc_bytes));

    /* Set all status to RUNNING, ticks to max_iters (CHAOTIC default) */
    for (int i = 0; i < batch_size; i++) {
        h_init_status[i] = STATUS_RUNNING;
        h_init_ticks[i]  = max_iters;
    }
    HIP_CHECK(hipMemcpy(d_status, h_init_status, meta_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ticks,  h_init_ticks,  meta_bytes, hipMemcpyHostToDevice));
    free(h_init_status); h_init_status = nullptr;
    free(h_init_ticks);  h_init_ticks  = nullptr;

    /* Ping-pong pointers */
    src = d_grid_A;
    dst = d_grid_B;

    /* ── Main iteration loop ───────────────────────────────────────────────── */
    for (int iter = 0; iter < max_iters && !all_done; iter++) {

        /* Kernel 1: CA step */
        hipLaunchKernelGGL(ca_step_kernel,
            grid_2d, block_2d, 0, 0,
            src, dst, d_delta, d_status, grid_w, grid_cells);

        /* Kernel 2: Convergence check */
        hipLaunchKernelGGL(convergence_check_kernel,
            dim3(conv_blocks), dim3(conv_threads), 0, 0,
            d_delta, d_status, d_ticks,
            grid_cells, iter, stability_threshold);

        /* Kernel 3: Role compute + ring write (from iter >= OSC_START) */
        if (iter >= OSC_START) {
            int ring_slot = (iter - OSC_START) % OSC_WINDOW;
            hipLaunchKernelGGL(role_compute_kernel,
                grid_2d, block_2d, 0, 0,
                dst, d_role_ring, d_status,
                grid_w, grid_cells, ring_slot);
        }

        /* Kernel 4+5: Oscillation detection (iter > 50, every 10 iters) */
        if (iter > 50 && (iter % 10) == 0) {
            int oldest_slot = (iter - 60) % OSC_WINDOW;
            hipLaunchKernelGGL(oscillation_detect_kernel,
                grid_2d, block_2d, osc_smem, 0,
                d_role_ring, d_osc, d_status,
                grid_w, grid_cells, oldest_slot);

            hipLaunchKernelGGL(oscillation_finalize_kernel,
                dim3(conv_blocks), dim3(conv_threads), 0, 0,
                d_osc, d_status, d_ticks,
                grid_cells, iter);
        }

        /* Swap ping-pong */
        float* tmp = src;
        src = dst;
        dst = tmp;

        /* Poll for all-done every 50 iterations */
        if ((iter % 50) == 49 || iter == max_iters - 1) {
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(h_poll_status, d_status, meta_bytes,
                                hipMemcpyDeviceToHost));
            all_done = true;
            for (int i = 0; i < batch_size; i++) {
                if (h_poll_status[i] == STATUS_RUNNING) {
                    all_done = false;
                    break;
                }
            }
        }
    }

    /* Final sync */
    HIP_CHECK(hipDeviceSynchronize());

    /* `src` points to the latest frame (after last swap) */
    HIP_CHECK(hipMemcpy(frames_out, src, grid_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ticks_out,  d_ticks,  meta_bytes, hipMemcpyDeviceToHost));

    /* Map status: RUNNING → CHAOTIC for output */
    HIP_CHECK(hipMemcpy(h_poll_status, d_status, meta_bytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < batch_size; i++) {
        states_out[i] = (h_poll_status[i] == STATUS_RUNNING)
                        ? STATUS_CHAOTIC : h_poll_status[i];
    }

cleanup:
    if (h_poll_status)  free(h_poll_status);
    if (h_init_status)  free(h_init_status);
    if (h_init_ticks)   free(h_init_ticks);
    if (d_grid_A)    (void)hipFree(d_grid_A);
    if (d_grid_B)    (void)hipFree(d_grid_B);
    if (d_role_ring) (void)hipFree(d_role_ring);
    if (d_delta)     (void)hipFree(d_delta);
    if (d_status)    (void)hipFree(d_status);
    if (d_ticks)     (void)hipFree(d_ticks);
    if (d_osc)       (void)hipFree(d_osc);

    return retval;
}


/**
 * ca_evolve_single_v2 — convenience wrapper for a single frame.
 */
int ca_evolve_single_v2(
    const float* frame_in,
    float*       frame_out,
    int*         ticks_out,
    int*         state_out,
    int          grid_w,
    int          max_iters,
    float        stability_threshold)
{
    return ca_evolve_batch_v2(
        frame_in, frame_out, ticks_out, state_out,
        1, grid_w, max_iters, stability_threshold);
}

}  /* extern "C" */
