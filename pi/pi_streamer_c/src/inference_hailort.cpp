// HailoRT C++ shim that exports the C ABI declared in
// include/pi_streamer/inference.h. Only compiled on the Pi when CMake is
// invoked with -DPI_USE_REAL_INFERENCE=ON and HailoRT is found. Host
// builds use src/inference_mock.c instead.
//
// Why C++ and not C:
//   On the Hailo-10H chip, the legacy HailoRT vstream C API
//   (hailo_create_configure_params + hailo_vstream_write/read_raw_buffer)
//   is NOT supported. HailoRT returns HAILO_NOT_IMPLEMENTED at
//   create_configure_params time with the message "Did you try calling
//   `create_configure_params` on H10? If so, use InferModel instead".
//   The async InferModel API is only exposed in <hailo/infer_model.hpp>,
//   which is a C++ header. This shim mirrors the Python reference in
//   pi/pi_streamer/inference.py that uses the same API from pyhailort.
//
// Pipeline overview (per model):
//   1. pi_vdevice_create -> hailort::VDevice::create() (one shared VDevice
//      for all 3 models so the HailoRT scheduler can time-slice between
//      them on the single /dev/hailo0).
//   2. pi_infer_model_create:
//        a. vdevice->create_infer_model(hef_path) -> shared_ptr<InferModel>
//        b. model->configure() -> ConfiguredInferModel
//        c. cfg_model.set_scheduler_priority(cfg->scheduler_priority)
//        d. Query input/output frame sizes and names via
//           model->input() / model->outputs(), and allocate page-aligned
//           I/O buffers (the API requires PAGE_SIZE alignment for zero-copy).
//        e. Spawn a worker thread that runs ConfiguredInferModel::run_async
//           whenever pi_infer_submit delivers a new frame.
//   3. pi_infer_submit copies the payload into the input buffer, flips a
//      pending flag, and signals the worker via a condition variable. The
//      worker builds a Bindings, runs the job async, waits up to 1 s for
//      completion, then publishes the raw output pointer back to the
//      main thread via a one-slot "has_result" latch.
//   4. pi_infer_poll returns the latest completed result (if any) without
//      blocking — matching the one-slot semantics of inference_mock.c.
//   5. pi_infer_model_destroy stops the worker, joins it, and frees the
//      aligned buffers. The unique_ptr/shared_ptr fields clean up the
//      HailoRT objects automatically.
//
// Synchronisation model is a single-slot SPSC between the main thread (one
// producer, one consumer because each model has its own worker) and the
// worker thread: one `pending_submit` flag and one `has_result` flag, both
// guarded by a std::mutex + condition_variable. No dynamic allocation on
// the hot path after create().
//
// Primary references:
//   - HailoRT headers on the Pi: /usr/include/hailo/infer_model.hpp,
//     vdevice.hpp, expected.hpp, buffer.hpp.
//   - Python reference (same API via pyhailort):
//     pi/pi_streamer/inference.py (HailoPoseInference class).
//   - No DTU atomic notes are cited.

extern "C" {
#include "pi_streamer/inference.h"
#include "pi_streamer/logger.h"
#include "pi_streamer/postprocess.h"
}

#include <hailo/hailort.hpp>
#include <hailo/hef.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/buffer.hpp>
#include <hailo/expected.hpp>

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Hailo I/O buffers must be aligned to PAGE_SIZE and their allocated length
// must also be a multiple of PAGE_SIZE, per the note on
// ConfiguredInferModel::Bindings::InferStream::set_buffer in infer_model.hpp.
// We round up to 4096 (Linux default) which matches sysconf(_SC_PAGESIZE) on
// the Pi 5.
constexpr size_t kPageSize = 4096;

size_t round_up_to_page(size_t n) {
    return (n + kPageSize - 1u) & ~(kPageSize - 1u);
}

// Allocate a buffer that is PAGE_SIZE aligned and whose length is a
// multiple of PAGE_SIZE. Returns nullptr on failure. The `*rounded_size`
// out param receives the actually-allocated size so callers can zero-fill
// the tail bytes.
uint8_t *alloc_aligned(size_t wanted, size_t *rounded_size) {
    const size_t rounded = round_up_to_page(wanted);
    void *p = nullptr;
    if (posix_memalign(&p, kPageSize, rounded) != 0 || p == nullptr) {
        return nullptr;
    }
    memset(p, 0, rounded);
    if (rounded_size) *rounded_size = rounded;
    return static_cast<uint8_t *>(p);
}

// Short, 15-char-max worker thread names. pthread_setname_np truncates
// silently at 15 characters, so we pre-truncate for clarity.
const char *worker_thread_name(pi_infer_kind_t k) {
    switch (k) {
        case PI_INFER_POSE:   return "pi-hailo-pose";
        case PI_INFER_OBJECT: return "pi-hailo-obj";
        case PI_INFER_HAND:   return "pi-hailo-hand";
        default:              return "pi-hailo-?";
    }
}

} // anonymous namespace

// ---- Opaque C types -------------------------------------------------------

struct pi_vdevice {
    std::unique_ptr<hailort::VDevice> vd;
};

// Per-output tensor metadata cached at configure time so the worker thread
// can bind buffers without re-querying the model on every request.
//
// `grid_h`, `grid_w`, `channels`, `dtype`, `qp_scale`, `qp_zp` are also
// captured at configure time because the pose postprocess needs them on
// every frame — querying the HEF each time would be absurd overhead and
// the values do not change during a run.
struct pi_infer_output_slot {
    std::string       name;
    size_t            size   = 0;   // exact frame size in bytes
    size_t            offset = 0;   // offset inside out_buf (page-aligned)
    int32_t           grid_h    = 0;
    int32_t           grid_w    = 0;
    int32_t           channels  = 0;
    pi_tensor_dtype_t dtype     = PI_TENSOR_DTYPE_U8;
    float             qp_scale  = 1.0f;
    float             qp_zp     = 0.0f;
};

struct pi_infer_model {
    pi_vdevice_t                        *vd_owner   = nullptr;
    pi_infer_model_config_t              cfg{};

    std::shared_ptr<hailort::InferModel> model;
    // ConfiguredInferModel has a public default constructor but is not
    // meant to be used uninitialised. We store it directly and reassign
    // via move after `model->configure()` returns.
    hailort::ConfiguredInferModel        cfg_model;

    // I/O buffer plan
    std::string                          in_name;
    size_t                               in_frame_size     = 0; // HailoRT frame size
    size_t                               in_buf_capacity   = 0; // page-rounded
    uint8_t                             *in_buf            = nullptr;

    std::vector<pi_infer_output_slot>    outs;
    size_t                               out_buf_capacity  = 0; // page-rounded
    uint8_t                             *out_buf           = nullptr;

    // Producer/consumer handshake (one-slot SPSC).
    std::mutex                           mtx;
    std::condition_variable              cv;
    bool                                 pending_submit    = false;
    bool                                 has_result        = false;
    uint64_t                             pending_frame_id  = 0;
    uint64_t                             result_frame_id   = 0;
    uint64_t                             result_latency_ns = 0;

    // Decoded detections from the most recent inference. The worker
    // thread runs model-specific postprocess right after wait() succeeds
    // and stores the result here so pi_infer_poll can hand it back as a
    // plain copy under mtx.
    uint8_t                              result_num_detections = 0;
    pi_detection_t                       result_detections[PI_MAX_DETECTIONS_PER_KIND];

    // Frame dimensions for letterbox reverse. Populated by the first
    // pi_infer_submit call — the caller (pipeline.c) passes them through
    // via a setter below rather than adding a second configure-time
    // argument that every existing caller would have to opt into.
    int32_t                              frame_width  = 0;
    int32_t                              frame_height = 0;

    // Worker thread lifecycle.
    std::atomic<bool>                    running{false};
    pthread_t                            worker            = 0;
    bool                                 worker_started    = false;

    // RAII so the `delete m` error paths in pi_infer_model_create don't
    // leak the posix_memalign'd I/O buffers. pi_infer_model_destroy sets
    // these to nullptr explicitly after free() so the destructor is a
    // no-op on the happy path.
    ~pi_infer_model() {
        if (in_buf)  { free(in_buf);  in_buf  = nullptr; }
        if (out_buf) { free(out_buf); out_buf = nullptr; }
    }
};

// ---- Internal helpers -----------------------------------------------------

namespace {

// Pretty-print a hailo_status into a short log message. HailoRT ships
// hailo_get_status_message but it lives in the C header — pull it via the
// extern "C" section already included at the top of this file.
const char *status_str(hailo_status s) {
    // hailort.h exposes hailo_get_status_message.
    const char *m = hailo_get_status_message(s);
    return m ? m : "(unknown)";
}

void *infer_worker_trampoline(void *arg);

} // anonymous namespace

// ---- VDevice --------------------------------------------------------------

extern "C" pi_vdevice_t *pi_vdevice_create(void) {
    auto exp = hailort::VDevice::create();
    if (!exp) {
        PI_ERROR("inference",
                 "hailort::VDevice::create() failed: status=%d (%s)",
                 static_cast<int>(exp.status()), status_str(exp.status()));
        return nullptr;
    }

    auto *vd = new (std::nothrow) pi_vdevice_t{};
    if (!vd) {
        PI_ERROR("inference", "out of memory allocating pi_vdevice_t");
        return nullptr;
    }
    vd->vd = exp.release();
    PI_INFO("inference", "HailoRT VDevice created (shared across models)");
    return vd;
}

extern "C" void pi_vdevice_destroy(pi_vdevice_t *vd) {
    // unique_ptr tears the VDevice down on delete.
    delete vd;
}

// ---- Worker thread --------------------------------------------------------

namespace {

void worker_run_one(pi_infer_model_t *m) {
    // Build a fresh Bindings every request. The HailoRT API allows Bindings
    // reuse, but constructing a new one here keeps the submit/poll latching
    // simple and avoids the per-frame mutation of a shared object.
    auto bindings_exp = m->cfg_model.create_bindings();
    if (!bindings_exp) {
        PI_WARN("inference",
                "[%s] create_bindings failed: status=%d (%s)",
                pi_infer_kind_name(m->cfg.kind),
                static_cast<int>(bindings_exp.status()),
                status_str(bindings_exp.status()));
        return;
    }
    auto bindings = bindings_exp.release();

    // Bind the input buffer. Bindings::InferStream is not default-
    // constructible, so we consume the Expected<> directly with release().
    {
        auto in_exp = bindings.input(m->in_name);
        if (!in_exp) {
            PI_WARN("inference",
                    "[%s] bindings.input('%s') failed: status=%d (%s)",
                    pi_infer_kind_name(m->cfg.kind), m->in_name.c_str(),
                    static_cast<int>(in_exp.status()),
                    status_str(in_exp.status()));
            return;
        }
        const hailo_status s = in_exp.release().set_buffer(
            hailort::MemoryView(m->in_buf, m->in_frame_size));
        if (s != HAILO_SUCCESS) {
            PI_WARN("inference",
                    "[%s] input set_buffer failed: status=%d (%s)",
                    pi_infer_kind_name(m->cfg.kind),
                    static_cast<int>(s), status_str(s));
            return;
        }
    }

    // Bind every output buffer.
    for (const auto &slot : m->outs) {
        auto out_exp = bindings.output(slot.name);
        if (!out_exp) {
            PI_WARN("inference",
                    "[%s] bindings.output('%s') failed: status=%d (%s)",
                    pi_infer_kind_name(m->cfg.kind), slot.name.c_str(),
                    static_cast<int>(out_exp.status()),
                    status_str(out_exp.status()));
            return;
        }
        const hailo_status s = out_exp.release().set_buffer(
            hailort::MemoryView(m->out_buf + slot.offset, slot.size));
        if (s != HAILO_SUCCESS) {
            PI_WARN("inference",
                    "[%s] output '%s' set_buffer failed: status=%d (%s)",
                    pi_infer_kind_name(m->cfg.kind), slot.name.c_str(),
                    static_cast<int>(s), status_str(s));
            return;
        }
    }

    // Make sure the async pipeline has room for at least one more frame.
    // Matches the Python reference's wait_for_async_ready(5000).
    {
        const hailo_status s =
            m->cfg_model.wait_for_async_ready(std::chrono::milliseconds(5000), 1);
        if (s != HAILO_SUCCESS) {
            PI_WARN("inference",
                    "[%s] wait_for_async_ready failed: status=%d (%s)",
                    pi_infer_kind_name(m->cfg.kind),
                    static_cast<int>(s), status_str(s));
            return;
        }
    }

    // Submit and wait. We issue a single-shot synchronous wait here so
    // the buffer lifetime is trivially bounded by the worker loop.
    const auto t0 = std::chrono::steady_clock::now();

    auto job_exp = m->cfg_model.run_async(bindings);
    if (!job_exp) {
        PI_WARN("inference",
                "[%s] run_async failed: status=%d (%s)",
                pi_infer_kind_name(m->cfg.kind),
                static_cast<int>(job_exp.status()),
                status_str(job_exp.status()));
        return;
    }
    auto job = job_exp.release();

    const hailo_status wait_s = job.wait(std::chrono::milliseconds(5000));
    if (wait_s != HAILO_SUCCESS) {
        PI_WARN("inference",
                "[%s] job.wait failed: status=%d (%s)",
                pi_infer_kind_name(m->cfg.kind),
                static_cast<int>(wait_s), status_str(wait_s));
        return;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t latency_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    // Model-specific postprocess. Runs on the worker thread so the raw
    // tensor bytes (still alive inside m->out_buf until the next submit)
    // can be consumed without a race, and so decode cost never hits the
    // encoder hot path.
    uint8_t        n_det = 0;
    pi_detection_t dets[PI_MAX_DETECTIONS_PER_KIND];
    if (m->cfg.kind == PI_INFER_POSE) {
        pi_postprocess_pose_cfg_t pcfg = {};
        pcfg.input_size      = (int32_t)m->cfg.input_width;
        pcfg.reg_max         = 16;
        pcfg.frame_width     = (m->cfg.frame_width > 0)
                               ? (int32_t)m->cfg.frame_width
                               : (int32_t)m->cfg.input_width;
        pcfg.frame_height    = (m->cfg.frame_height > 0)
                               ? (int32_t)m->cfg.frame_height
                               : (int32_t)m->cfg.input_height;
        pcfg.score_threshold = 0.25f;
        pcfg.iou_threshold   = 0.7f;
        pcfg.num_tensors     = (m->outs.size() > 9) ? 9 : m->outs.size();
        for (size_t i = 0; i < pcfg.num_tensors; i++) {
            const auto &slot = m->outs[i];
            pcfg.tensors[i] = pi_tensor_view_t{
                /* data */       m->out_buf + slot.offset,
                /* dtype */      slot.dtype,
                /* scale */      slot.qp_scale,
                /* zero_point */ slot.qp_zp,
                /* grid_h */     slot.grid_h,
                /* grid_w */     slot.grid_w,
                /* channels */   slot.channels,
            };
        }
        const size_t dn = pi_postprocess_pose_decode(
            &pcfg, dets, PI_MAX_DETECTIONS_PER_KIND);
        n_det = (dn > 255u) ? 255u : (uint8_t)dn;
    }
    // OBJECT and HAND postprocess are follow-ups — the overlay will
    // show pose boxes only for the first iteration, which is enough to
    // verify the end-to-end wiring works on real hardware.

    // Publish result back to the main thread.
    {
        std::lock_guard<std::mutex> lk(m->mtx);
        m->has_result        = true;
        m->result_frame_id   = m->pending_frame_id;
        m->result_latency_ns = latency_ns;
        m->result_num_detections = n_det;
        if (n_det > 0) {
            memcpy(m->result_detections, dets,
                   (size_t)n_det * sizeof(pi_detection_t));
        }
    }
    // Note: we do not notify_one here because pi_infer_poll is non-blocking.
}

void *infer_worker_trampoline(void *arg) {
    auto *m = static_cast<pi_infer_model_t *>(arg);
    // Best-effort thread name (15-char limit).
    pthread_setname_np(pthread_self(), worker_thread_name(m->cfg.kind));

    while (m->running.load(std::memory_order_acquire)) {
        // Wait for a submission or a shutdown request.
        {
            std::unique_lock<std::mutex> lk(m->mtx);
            m->cv.wait(lk, [m]() {
                return m->pending_submit ||
                       !m->running.load(std::memory_order_acquire);
            });
            if (!m->running.load(std::memory_order_acquire)) {
                return nullptr;
            }
            // Consume the submit flag; run_one will re-use in_buf.
            m->pending_submit = false;
        }

        worker_run_one(m);
    }
    return nullptr;
}

} // anonymous namespace

// ---- InferModel lifecycle -------------------------------------------------

extern "C" pi_infer_model_t *pi_infer_model_create(pi_vdevice_t                   *vd,
                                                   const pi_infer_model_config_t *cfg) {
    if (!vd || !vd->vd || !cfg || !cfg->hef_path) {
        return nullptr;
    }
    if (static_cast<unsigned>(cfg->kind) >= PI_INFER__COUNT) {
        return nullptr;
    }

    auto *m = new (std::nothrow) pi_infer_model_t{};
    if (!m) {
        PI_ERROR("inference", "out of memory allocating pi_infer_model_t");
        return nullptr;
    }
    m->vd_owner = vd;
    m->cfg      = *cfg;

    // 1. Create InferModel from HEF file.
    auto model_exp = vd->vd->create_infer_model(cfg->hef_path);
    if (!model_exp) {
        PI_ERROR("inference",
                 "[%s] create_infer_model('%s') failed: status=%d (%s)",
                 pi_infer_kind_name(cfg->kind), cfg->hef_path,
                 static_cast<int>(model_exp.status()),
                 status_str(model_exp.status()));
        delete m;
        return nullptr;
    }
    m->model = model_exp.release();

    // Keep a fixed batch size of 1 — matches the Python reference and keeps
    // the scheduler's burst window tight so all 3 models can interleave.
    m->model->set_batch_size(1);

    // 2. Configure the model. The resulting ConfiguredInferModel is stored
    //    by value; ConfiguredInferModel is move-assignable so this works
    //    without std::optional/unique_ptr gymnastics.
    auto cfg_exp = m->model->configure();
    if (!cfg_exp) {
        PI_ERROR("inference",
                 "[%s] model->configure() failed: status=%d (%s)",
                 pi_infer_kind_name(cfg->kind),
                 static_cast<int>(cfg_exp.status()),
                 status_str(cfg_exp.status()));
        delete m;
        return nullptr;
    }
    m->cfg_model = cfg_exp.release();

    // 3. Set scheduler priority. HailoRT docs: larger number = higher prio;
    //    the pipeline.c config passes POSE=18, OBJECT=17, HAND=15, mirroring
    //    the Python reference's HAILO_SCHEDULER_PRIORITY_{POSE,OBJECT,HAND}.
    {
        const uint8_t prio = static_cast<uint8_t>(cfg->scheduler_priority);
        const hailo_status s = m->cfg_model.set_scheduler_priority(prio);
        if (s != HAILO_SUCCESS) {
            PI_WARN("inference",
                    "[%s] set_scheduler_priority(%u) failed: status=%d (%s)",
                    pi_infer_kind_name(cfg->kind), static_cast<unsigned>(prio),
                    static_cast<int>(s), status_str(s));
            // Non-fatal — HailoRT still runs the model, just without the
            // explicit priority hint.
        }
    }

    // 4. Query input stream metadata. We assume a single input, which is
    //    true for all three HEFs in the pi_streamer pipeline.
    const auto &input_names = m->model->get_input_names();
    if (input_names.size() != 1) {
        PI_ERROR("inference",
                 "[%s] expected exactly 1 input stream, got %zu",
                 pi_infer_kind_name(cfg->kind), input_names.size());
        delete m;
        return nullptr;
    }
    m->in_name = input_names.front();

    auto in_stream_exp = m->model->input(m->in_name);
    if (!in_stream_exp) {
        PI_ERROR("inference",
                 "[%s] model->input('%s') failed: status=%d (%s)",
                 pi_infer_kind_name(cfg->kind), m->in_name.c_str(),
                 static_cast<int>(in_stream_exp.status()),
                 status_str(in_stream_exp.status()));
        delete m;
        return nullptr;
    }
    m->in_frame_size = in_stream_exp.release().get_frame_size();

    m->in_buf = alloc_aligned(m->in_frame_size, &m->in_buf_capacity);
    if (!m->in_buf) {
        PI_ERROR("inference",
                 "[%s] posix_memalign(%zu) for input buffer failed",
                 pi_infer_kind_name(cfg->kind), m->in_frame_size);
        delete m;
        return nullptr;
    }

    // 5. Query output streams. Lay them out back-to-back inside a single
    //    page-aligned buffer so we only make one allocation. Each slot's
    //    exact frame_size feeds set_buffer; any rounding slack inside
    //    out_buf_capacity is ignored by HailoRT.
    //
    // We also load the HEF separately here to read quantization + shape
    // metadata for each output tensor — the ConfiguredInferModel stream
    // handle only exposes `get_frame_size()` reliably, and the pose
    // postprocess step needs the (H, W, C) shape, the dtype, and the
    // (scale, zero_point) pair per tensor. Mirrors the Python reference's
    // HEF.get_output_vstream_infos() path in pi_streamer/inference.py:295.
    auto hef_exp = hailort::Hef::create(cfg->hef_path);
    if (!hef_exp) {
        PI_ERROR("inference",
                 "[%s] Hef::create('%s') failed: status=%d (%s)",
                 pi_infer_kind_name(cfg->kind), cfg->hef_path,
                 static_cast<int>(hef_exp.status()),
                 status_str(hef_exp.status()));
        delete m;
        return nullptr;
    }
    auto hef = hef_exp.release();
    auto vinfos_exp = hef.get_output_vstream_infos();
    if (!vinfos_exp) {
        PI_ERROR("inference",
                 "[%s] Hef::get_output_vstream_infos failed: status=%d (%s)",
                 pi_infer_kind_name(cfg->kind),
                 static_cast<int>(vinfos_exp.status()),
                 status_str(vinfos_exp.status()));
        delete m;
        return nullptr;
    }
    const auto vinfos = vinfos_exp.release();

    const auto &output_names = m->model->get_output_names();
    if (output_names.empty()) {
        PI_ERROR("inference", "[%s] model has 0 outputs",
                 pi_infer_kind_name(cfg->kind));
        delete m;
        return nullptr;
    }

    size_t total = 0;
    m->outs.reserve(output_names.size());
    for (const auto &name : output_names) {
        auto out_exp = m->model->output(name);
        if (!out_exp) {
            PI_ERROR("inference",
                     "[%s] model->output('%s') failed: status=%d (%s)",
                     pi_infer_kind_name(cfg->kind), name.c_str(),
                     static_cast<int>(out_exp.status()),
                     status_str(out_exp.status()));
            delete m;
            return nullptr;
        }
        const size_t sz = out_exp.release().get_frame_size();

        pi_infer_output_slot slot;
        slot.name   = name;
        slot.size   = sz;
        slot.offset = total;

        // Look up matching vstream info by name. HailoRT returns a flat
        // vector so we linear-search — 9 outputs at most, no sweat.
        for (const auto &vi : vinfos) {
            if (name == vi.name) {
                slot.grid_h   = (int32_t)vi.shape.height;
                slot.grid_w   = (int32_t)vi.shape.width;
                slot.channels = (int32_t)vi.shape.features;
                slot.qp_scale = vi.quant_info.qp_scale;
                slot.qp_zp    = vi.quant_info.qp_zp;
                slot.dtype    = (vi.format.type == HAILO_FORMAT_TYPE_UINT16)
                                ? PI_TENSOR_DTYPE_U16
                                : PI_TENSOR_DTYPE_U8;
                break;
            }
        }

        PI_DEBUG("inference",
                 "[%s] output '%s' shape=%dx%dx%d dtype=%s scale=%.6f zp=%.1f sz=%zu",
                 pi_infer_kind_name(cfg->kind), name.c_str(),
                 slot.grid_h, slot.grid_w, slot.channels,
                 (slot.dtype == PI_TENSOR_DTYPE_U16) ? "u16" : "u8",
                 (double)slot.qp_scale, (double)slot.qp_zp, sz);

        m->outs.push_back(std::move(slot));
        // Each output buffer section must itself be page-aligned to satisfy
        // the set_buffer requirement, so we round every slot's size up.
        total += round_up_to_page(sz);
    }

    m->out_buf = alloc_aligned(total, &m->out_buf_capacity);
    if (!m->out_buf) {
        PI_ERROR("inference",
                 "[%s] posix_memalign(%zu) for output buffer failed",
                 pi_infer_kind_name(cfg->kind), total);
        delete m;
        return nullptr;
    }

    // 6. Spawn the worker thread. It starts idle and blocks on cv until
    //    pi_infer_submit posts the first frame.
    m->running.store(true, std::memory_order_release);
    if (pthread_create(&m->worker, nullptr, infer_worker_trampoline, m) != 0) {
        PI_ERROR("inference", "[%s] pthread_create for worker failed",
                 pi_infer_kind_name(cfg->kind));
        m->running.store(false, std::memory_order_release);
        delete m;
        return nullptr;
    }
    m->worker_started = true;

    PI_INFO("inference",
            "[%s] configured: hef='%s' in='%s' in_size=%zu outputs=%zu prio=%d",
            pi_infer_kind_name(cfg->kind), cfg->hef_path,
            m->in_name.c_str(), m->in_frame_size,
            m->outs.size(), cfg->scheduler_priority);
    return m;
}

extern "C" void pi_infer_model_destroy(pi_infer_model_t *model) {
    if (!model) return;

    // 1. Tell the worker to exit and unblock its cv.
    if (model->worker_started) {
        {
            std::lock_guard<std::mutex> lk(model->mtx);
            model->running.store(false, std::memory_order_release);
        }
        model->cv.notify_all();
        pthread_join(model->worker, nullptr);
    }

    // 2. Free aligned I/O buffers.
    free(model->in_buf);
    free(model->out_buf);
    model->in_buf  = nullptr;
    model->out_buf = nullptr;

    // 3. ConfiguredInferModel / InferModel / shared_ptr destructors run
    //    automatically when `delete model` returns.
    delete model;
}

// ---- Submit / poll --------------------------------------------------------

extern "C" int pi_infer_submit(pi_infer_model_t *model,
                               const uint8_t    *input,
                               size_t            input_size,
                               uint64_t          frame_id) {
    if (!model || !input || input_size == 0) return -1;
    if (!model->running.load(std::memory_order_acquire)) return -1;

    {
        std::lock_guard<std::mutex> lk(model->mtx);
        // One-slot queue: refuse new submissions while a previous job is
        // still in flight OR while the previous result hasn't been polled.
        // This matches inference_mock.c's back-pressure semantics so the
        // pipeline behaves identically on host and Pi.
        if (model->pending_submit || model->has_result) {
            return -1;
        }

        // Copy up to in_frame_size bytes into the pinned input buffer. If
        // the caller passes a larger buffer we silently truncate (matches
        // the libcamera capture path, which feeds full-frame BGR); if the
        // caller passes smaller we leave the tail at its zero-initialised
        // value, which is safe because the worker binds in_frame_size
        // exactly and HailoRT does not read beyond that.
        const size_t copy_n = (input_size < model->in_frame_size)
                              ? input_size : model->in_frame_size;
        memcpy(model->in_buf, input, copy_n);
        if (copy_n < model->in_frame_size) {
            memset(model->in_buf + copy_n, 0, model->in_frame_size - copy_n);
        }

        model->pending_submit   = true;
        model->pending_frame_id = frame_id;
    }
    model->cv.notify_one();
    return 0;
}

extern "C" int pi_infer_poll(pi_infer_model_t *model, pi_infer_result_t *out) {
    if (!model || !out) return -1;

    std::lock_guard<std::mutex> lk(model->mtx);
    if (!model->has_result) return -1;

    // Zero the result first so the detections[] tail is never uninit
    // on the seqlock publish path (matches inference_mock.c semantics).
    memset(out, 0, sizeof *out);

    out->frame_id        = model->result_frame_id;
    out->kind            = model->cfg.kind;
    out->raw_output      = model->out_buf;
    out->raw_output_size = model->out_buf_capacity;
    out->latency_ns      = model->result_latency_ns;

    out->num_detections = model->result_num_detections;
    if (out->num_detections > 0) {
        memcpy(out->detections, model->result_detections,
               (size_t)out->num_detections * sizeof(pi_detection_t));
    }

    // Clear the latch so the next submit is accepted.
    model->has_result = false;
    return 0;
}

extern "C" const char *pi_infer_kind_name(pi_infer_kind_t k) {
    switch (k) {
        case PI_INFER_POSE:   return "pose";
        case PI_INFER_OBJECT: return "object";
        case PI_INFER_HAND:   return "hand";
        case PI_INFER__COUNT: return "<invalid>";
    }
    return "<unknown>";
}
