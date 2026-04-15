// libcamera C++ shim that exports the C ABI declared in
// include/pi_streamer/camera.h. This file is ONLY compiled on the Pi when
// CMake is invoked with -DPI_USE_REAL_CAMERA=ON. Host builds use
// src/camera_mock.c instead.
//
// Pipeline overview:
//   1. Open the IMX708 wide camera via libcamera::CameraManager.
//   2. Configure a single VideoRecording stream as planar YUV420 at the
//      requested resolution (default 1280x720), with bufferCount=8.
//   3. Force the IMX708 wide-binned 2304x1296 sensor mode by hinting the
//      raw stream size, and pin ScalerCrop to the full sensor rectangle so
//      the wide field of view is preserved (no center crop).
//   4. Allocate all FrameBuffers via FrameBufferAllocator and mmap each
//      DMA-BUF plane ONCE up front (zero-copy on every subsequent capture).
//   5. Build one Request per FrameBuffer and queue them all in pi_camera_start.
//   6. Wire up Camera::requestCompleted -> a small SPSC ring of completed
//      slots, and notify a condition variable so pi_camera_capture can wake.
//   7. pi_camera_release re-uses the request (Request::ReuseBuffers) and
//      re-queues it so libcamera keeps cycling the same buffer pool.
//
// Note on MappedFrameBuffer: in libcamera 0.7.0 (Debian 13 packaging) the
// libcamera::MappedFrameBuffer helper is in an internal header that is not
// installed publicly. We therefore mmap() the DMA-BUF plane fds ourselves.
// rpicam-apps does the same (see core/libcamera_app.cpp). Multiple planes
// can share one underlying fd at different offsets, so we mmap once per
// unique fd and resolve plane base pointers via (offset, length).
//
// Primary references:
//   - libcamera API:        https://libcamera.org/api-html/
//   - rpicam-apps source:   github.com/raspberrypi/rpicam-apps
//   - Python equivalent:    pi/pi_streamer/camera.py (PiCamera class)

extern "C" {
#include "pi_streamer/camera.h"
#include "pi_streamer/ring_buffer.h"
#include "pi_streamer/logger.h"
}

#include <libcamera/libcamera.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/geometry.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// IMX708 full sensor rectangle. Used for ScalerCrop to preserve full wide
// FOV (matches pi/pi_streamer/camera.py: IMX708_FULL_SENSOR).
constexpr int kImx708FullX      = 0;
constexpr int kImx708FullY      = 0;
constexpr int kImx708FullWidth  = 4608;
constexpr int kImx708FullHeight = 2592;

// Number of buffers in the camera pool. Must be >= 4 to hide ISP latency
// and small enough to keep memory pressure reasonable.
constexpr unsigned int kBufferCount = 8;

// SPSC ring capacity for completed slots. Must be a power of two and >=
// kBufferCount so the producer never drops a completion.
constexpr size_t kCompletedRingCapacity = 16;

// One mmap'd DMA-BUF region. Multiple FrameBuffer planes may share the
// same fd at different offsets, so we deduplicate mappings per fd.
struct MappedRegion {
    int      fd     = -1;
    void    *addr   = MAP_FAILED;
    size_t   length = 0;
};

// Per-FrameBuffer cache of plane base pointers and stride metadata. Built
// once at start-up, then read directly on every capture.
struct MappedBuffer {
    libcamera::FrameBuffer *buffer    = nullptr;
    const uint8_t          *y_plane   = nullptr;
    const uint8_t          *u_plane   = nullptr;
    const uint8_t          *v_plane   = nullptr;
    size_t                  y_size    = 0;
    size_t                  u_size    = 0;
    size_t                  v_size    = 0;
    uint32_t                y_stride  = 0;
    uint32_t                uv_stride = 0;
};

// One slot pushed onto the completed ring per requestCompleted signal.
// Lives on the heap; deleted in pi_camera_release after the request has
// been re-queued.
struct CompletedSlot {
    libcamera::Request *request      = nullptr;
    MappedBuffer       *mapped       = nullptr;
    uint64_t            timestamp_ns = 0;
    uint64_t            frame_id     = 0;
};

} // anonymous namespace

// ---- pi_camera opaque type --------------------------------------------------

struct pi_camera {
    pi_camera_config_t cfg{};

    std::unique_ptr<libcamera::CameraManager>           cm;
    std::shared_ptr<libcamera::Camera>                  cam;
    std::unique_ptr<libcamera::CameraConfiguration>     config;
    std::unique_ptr<libcamera::FrameBufferAllocator>    allocator;
    libcamera::Stream                                  *stream = nullptr;

    // Owned buffers/requests. Lifetime tied to this pi_camera.
    std::vector<std::unique_ptr<libcamera::Request>>    requests;
    std::vector<MappedBuffer>                           mapped_buffers;
    std::vector<MappedRegion>                           mmap_regions;

    // Completion plumbing.
    pi_ring_buffer_t                                   *completed = nullptr;
    std::mutex                                          completed_mtx;
    std::condition_variable                             completed_cv;

    std::atomic<bool>                                   running{false};
    std::atomic<bool>                                   acquired{false};
    std::atomic<bool>                                   started{false};
    std::atomic<uint64_t>                               frame_counter{0};

    // Counts in-flight requestCompleted lambda invocations. teardown()
    // spins on this to drop to zero AFTER disconnect() so no lambda can
    // still be reading `mapped_buffers` / `completed` when we clear
    // them. The lambda bumps it on entry and decrements on return.
    std::atomic<int>                                    signal_inflight{0};
};

// ---- internal helpers -------------------------------------------------------

namespace {

// Look up an existing mmap region by fd, or create one by mmap'ing the
// whole FD length. Returns the base pointer or nullptr on failure.
uint8_t *mapPlaneRegion(pi_camera *cam, int fd, size_t length) {
    for (auto &region : cam->mmap_regions) {
        if (region.fd == fd) {
            // Same fd may back a longer FrameBuffer than what an individual
            // plane reports — extend the existing mapping if needed.
            if (length <= region.length) {
                return static_cast<uint8_t *>(region.addr);
            }
            // The first FrameBuffer set the canonical length. If a later
            // plane needs more, that means the dmabuf has multiple planes
            // we missed — log and refuse rather than silently corrupt.
            PI_ERROR("camera",
                     "fd %d already mapped at length %zu, plane needs %zu",
                     fd, region.length, length);
            return nullptr;
        }
    }

    void *addr = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        PI_ERROR("camera", "mmap fd=%d length=%zu failed", fd, length);
        return nullptr;
    }
    cam->mmap_regions.push_back({fd, addr, length});
    return static_cast<uint8_t *>(addr);
}

// Compute the total byte length needed for a FrameBuffer's mapping on a
// given fd: the maximum of (plane.offset + plane.length) over all planes
// that share that fd.
size_t fdLengthForBuffer(libcamera::FrameBuffer *buffer, int fd) {
    size_t needed = 0;
    for (const auto &plane : buffer->planes()) {
        if (plane.fd.get() != fd) continue;
        const size_t end = static_cast<size_t>(plane.offset) +
                           static_cast<size_t>(plane.length);
        if (end > needed) needed = end;
    }
    return needed;
}

// Populate one MappedBuffer entry from a freshly allocated FrameBuffer.
// Returns true on success.
bool mapFrameBuffer(pi_camera                *cam,
                    libcamera::FrameBuffer   *buffer,
                    const libcamera::StreamConfiguration &stream_cfg,
                    MappedBuffer             *out) {
    auto planes = buffer->planes();
    if (planes.size() < 1) {
        PI_ERROR("camera", "FrameBuffer has 0 planes");
        return false;
    }

    // libcamera may report YUV420 either as 3 distinct planes or as 1
    // contiguous plane covering Y+U+V. Handle both layouts.
    const uint32_t width    = stream_cfg.size.width;
    const uint32_t height   = stream_cfg.size.height;
    const uint32_t y_stride = stream_cfg.stride;
    const uint32_t uv_stride = y_stride / 2u;

    out->buffer    = buffer;
    out->y_stride  = y_stride;
    out->uv_stride = uv_stride;

    if (planes.size() >= 3) {
        // Three-plane layout: each plane has its own (fd, offset, length).
        for (size_t i = 0; i < 3; ++i) {
            const int    fd     = planes[i].fd.get();
            const size_t length = fdLengthForBuffer(buffer, fd);
            uint8_t     *base   = mapPlaneRegion(cam, fd, length);
            if (!base) return false;

            const uint8_t *p = base + planes[i].offset;
            switch (i) {
                case 0: out->y_plane = p; out->y_size = planes[i].length; break;
                case 1: out->u_plane = p; out->u_size = planes[i].length; break;
                case 2: out->v_plane = p; out->v_size = planes[i].length; break;
            }
        }
    } else {
        // Single-plane layout: derive U and V offsets from the canonical
        // I420 geometry. This is the path libcamera typically takes for
        // YUV420 on the Pi.
        const int    fd     = planes[0].fd.get();
        const size_t length = fdLengthForBuffer(buffer, fd);
        uint8_t     *base   = mapPlaneRegion(cam, fd, length);
        if (!base) return false;

        const size_t y_size  = static_cast<size_t>(y_stride)  * height;
        const size_t uv_size = static_cast<size_t>(uv_stride) * (height / 2u);

        out->y_plane = base + planes[0].offset;
        out->u_plane = out->y_plane + y_size;
        out->v_plane = out->u_plane + uv_size;
        out->y_size  = y_size;
        out->u_size  = uv_size;
        out->v_size  = uv_size;

        // Sanity: the combined Y+U+V must fit inside the reported plane
        // length. Refuse rather than read past the mapping.
        if (y_size + 2u * uv_size > planes[0].length) {
            PI_ERROR("camera",
                     "single-plane YUV420 too small: have %u, need %zu "
                     "(width=%u height=%u y_stride=%u)",
                     planes[0].length, y_size + 2u * uv_size,
                     width, height, y_stride);
            return false;
        }
    }

    return true;
}

// Tear down all owned camera state. Safe to call from create() partial
// failure paths and from destroy(). Idempotent.
void teardown(pi_camera *cam) noexcept {
    if (!cam) return;

    // stop() must run BEFORE disconnect() so libcamera drains all
    // in-flight requests (cancelled callbacks still fire but see
    // req->status() != RequestComplete and early-return). We then spin
    // briefly until signal_inflight reaches zero — this closes the
    // micro-window where a cancelled-status lambda is mid-return, its
    // `if (req->status() ...)` check has finished, but the bookkeeping
    // decrement hasn't. Without this wait, requests.clear() below could
    // destroy Request objects while a lambda still holds a `req *` on
    // its stack, resulting in use-after-free on the libcamera event
    // thread.
    if (cam->started.load()) {
        if (cam->cam) cam->cam->stop();
        cam->started = false;
    }

    // Disconnect the requestCompleted signal before releasing the camera
    // so no NEW callback can fire against a half-destroyed pi_camera.
    if (cam->cam) {
        cam->cam->requestCompleted.disconnect();
    }

    // Wait (bounded) for any IN-PROGRESS lambda invocation to return.
    // 100 spins × 1 ms = 100 ms budget. Signals are tiny (a few hundred
    // ns of work) so 1 ms is already vastly overkill, but we keep the
    // upper bound generous because teardown runs in the slow path.
    for (int i = 0; i < 100; i++) {
        if (cam->signal_inflight.load(std::memory_order_acquire) == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cam->requests.clear();

    if (cam->allocator && cam->stream) {
        cam->allocator->free(cam->stream);
    }
    cam->allocator.reset();

    for (auto &region : cam->mmap_regions) {
        if (region.addr != MAP_FAILED && region.length > 0) {
            munmap(region.addr, region.length);
        }
    }
    cam->mmap_regions.clear();
    cam->mapped_buffers.clear();

    if (cam->completed) {
        // Drain any leftover slots so we don't leak heap entries.
        void *slot = nullptr;
        while (pi_ring_pop(cam->completed, &slot) == 0) {
            delete static_cast<CompletedSlot *>(slot);
        }
        pi_ring_destroy(cam->completed);
        cam->completed = nullptr;
    }

    if (cam->cam && cam->acquired.load()) {
        cam->cam->release();
        cam->acquired = false;
    }
    cam->cam.reset();
    cam->config.reset();

    if (cam->cm) {
        cam->cm->stop();
        cam->cm.reset();
    }
}

} // anonymous namespace

// ---- C ABI ------------------------------------------------------------------

extern "C" pi_camera_t *pi_camera_create(const pi_camera_config_t *cfg) {
    if (!cfg || cfg->width == 0 || cfg->height == 0) {
        return nullptr;
    }
    // YUV420 needs even width/height so the chroma planes line up.
    if ((cfg->width & 1u) || (cfg->height & 1u)) {
        return nullptr;
    }

    auto *cam = new (std::nothrow) pi_camera{};
    if (!cam) return nullptr;
    cam->cfg = *cfg;

    cam->cm = std::make_unique<libcamera::CameraManager>();
    if (cam->cm->start() != 0) {
        PI_ERROR("camera", "CameraManager::start() failed");
        delete cam;
        return nullptr;
    }

    if (cam->cm->cameras().empty()) {
        PI_ERROR("camera", "no cameras detected by libcamera");
        teardown(cam);
        delete cam;
        return nullptr;
    }

    cam->cam = cam->cm->cameras().front();
    PI_INFO("camera", "selected camera id=%s", cam->cam->id().c_str());

    if (cam->cam->acquire() != 0) {
        PI_ERROR("camera", "Camera::acquire() failed for %s",
                 cam->cam->id().c_str());
        teardown(cam);
        delete cam;
        return nullptr;
    }
    cam->acquired = true;

    // Ask libcamera for a default VideoRecording configuration, then pin
    // it to YUV420 at the requested size.
    cam->config = cam->cam->generateConfiguration(
        {libcamera::StreamRole::VideoRecording});
    if (!cam->config || cam->config->size() == 0) {
        PI_ERROR("camera", "generateConfiguration returned empty config");
        teardown(cam);
        delete cam;
        return nullptr;
    }

    libcamera::StreamConfiguration &stream_cfg = cam->config->at(0);
    stream_cfg.pixelFormat = libcamera::formats::YUV420;
    stream_cfg.size        = libcamera::Size(cfg->width, cfg->height);
    stream_cfg.bufferCount = kBufferCount;

    const auto validation = cam->config->validate();
    if (validation == libcamera::CameraConfiguration::Invalid) {
        PI_ERROR("camera", "CameraConfiguration::validate() = Invalid");
        teardown(cam);
        delete cam;
        return nullptr;
    }
    if (stream_cfg.pixelFormat != libcamera::formats::YUV420) {
        // Stop condition: the encoder path depends on YUV420 zero-copy.
        // We deliberately refuse to fall back to a software BGR
        // conversion — that would defeat the whole pipeline design.
        PI_ERROR("camera",
                 "libcamera refused YUV420 on main stream (got %s) — "
                 "stopping rather than falling back to software conversion",
                 stream_cfg.pixelFormat.toString().c_str());
        teardown(cam);
        delete cam;
        return nullptr;
    }
    if (validation == libcamera::CameraConfiguration::Adjusted) {
        PI_WARN("camera",
                "configuration adjusted by libcamera: size=%ux%u stride=%u",
                stream_cfg.size.width, stream_cfg.size.height,
                stream_cfg.stride);
    }

    if (cam->cam->configure(cam->config.get()) != 0) {
        PI_ERROR("camera", "Camera::configure() failed");
        teardown(cam);
        delete cam;
        return nullptr;
    }
    cam->stream = stream_cfg.stream();

    // Allocate the shared FrameBuffer pool.
    cam->allocator = std::make_unique<libcamera::FrameBufferAllocator>(cam->cam);
    if (cam->allocator->allocate(cam->stream) < 0) {
        PI_ERROR("camera", "FrameBufferAllocator::allocate() failed");
        teardown(cam);
        delete cam;
        return nullptr;
    }

    const auto &buffers = cam->allocator->buffers(cam->stream);
    cam->mapped_buffers.reserve(buffers.size());
    cam->requests.reserve(buffers.size());

    // mmap each FrameBuffer once and build a Request that owns it.
    for (const auto &buffer : buffers) {
        MappedBuffer mb;
        if (!mapFrameBuffer(cam, buffer.get(), stream_cfg, &mb)) {
            teardown(cam);
            delete cam;
            return nullptr;
        }
        cam->mapped_buffers.push_back(mb);

        auto request = cam->cam->createRequest();
        if (!request) {
            PI_ERROR("camera", "Camera::createRequest() returned null");
            teardown(cam);
            delete cam;
            return nullptr;
        }
        if (request->addBuffer(cam->stream, buffer.get()) != 0) {
            PI_ERROR("camera", "Request::addBuffer() failed");
            teardown(cam);
            delete cam;
            return nullptr;
        }

        // Set per-request controls: lock the frame duration to the target
        // FPS and pin ScalerCrop to the full IMX708 sensor rectangle so
        // the wide field of view is preserved (matches Python reference).
        const int64_t frame_us = (cfg->fps > 0)
            ? static_cast<int64_t>(1'000'000 / cfg->fps)
            : 33'333; // 30 fps fallback
        const std::array<int64_t, 2> frame_limits{frame_us, frame_us};
        request->controls().set(libcamera::controls::FrameDurationLimits,
                                libcamera::Span<const int64_t, 2>(frame_limits));
        request->controls().set(libcamera::controls::ScalerCrop,
                                libcamera::Rectangle(kImx708FullX,
                                                     kImx708FullY,
                                                     kImx708FullWidth,
                                                     kImx708FullHeight));

        cam->requests.push_back(std::move(request));
    }

    cam->completed = pi_ring_create(kCompletedRingCapacity);
    if (!cam->completed) {
        PI_ERROR("camera", "pi_ring_create(%zu) failed", kCompletedRingCapacity);
        teardown(cam);
        delete cam;
        return nullptr;
    }

    // Wire up the completion signal. The lambda runs on libcamera's
    // internal thread, so we keep it short: find the matching MappedBuffer,
    // build a slot, push to the SPSC ring, and notify the consumer CV.
    //
    // Bracket the whole lambda body in signal_inflight inc/dec so
    // teardown() can wait for in-flight invocations to finish before
    // clearing requests. A small RAII guard keeps every exit path
    // (early return, exception) balanced.
    cam->cam->requestCompleted.connect(cam, [cam](libcamera::Request *req) {
        struct InflightGuard {
            pi_camera *c;
            explicit InflightGuard(pi_camera *p) : c(p) {
                c->signal_inflight.fetch_add(1, std::memory_order_acq_rel);
            }
            ~InflightGuard() {
                c->signal_inflight.fetch_sub(1, std::memory_order_acq_rel);
            }
        } guard{cam};

        if (req->status() != libcamera::Request::RequestComplete) {
            // Request was cancelled (e.g. during stop) — don't emit a slot.
            return;
        }

        const auto &request_buffers = req->buffers();
        if (request_buffers.empty()) return;
        libcamera::FrameBuffer *fb = request_buffers.begin()->second;

        MappedBuffer *mapped = nullptr;
        for (auto &mb : cam->mapped_buffers) {
            if (mb.buffer == fb) {
                mapped = &mb;
                break;
            }
        }
        if (!mapped) {
            PI_WARN("camera", "completed request for unknown FrameBuffer");
            return;
        }

        const auto sensor_ts = req->metadata().get(
            libcamera::controls::SensorTimestamp);
        const uint64_t ts_ns = sensor_ts.has_value()
            ? static_cast<uint64_t>(*sensor_ts)
            : 0u;

        auto *slot = new (std::nothrow) CompletedSlot{
            req,
            mapped,
            ts_ns,
            cam->frame_counter.fetch_add(1, std::memory_order_relaxed),
        };
        if (!slot) {
            PI_ERROR("camera", "out of memory allocating CompletedSlot");
            return;
        }

        if (pi_ring_push(cam->completed, slot) != 0) {
            // Ring full — drop the frame and re-queue the request
            // immediately so libcamera doesn't stall. This should not
            // happen in steady state because kCompletedRingCapacity >=
            // kBufferCount.
            PI_WARN("camera", "completion ring full, dropping frame_id=%llu",
                    static_cast<unsigned long long>(slot->frame_id));
            delete slot;
            req->reuse(libcamera::Request::ReuseBuffers);
            cam->cam->queueRequest(req);
            return;
        }

        cam->completed_cv.notify_one();
    });

    PI_INFO("camera",
            "configured: %ux%u YUV420 stride=%u buffers=%zu",
            stream_cfg.size.width, stream_cfg.size.height,
            stream_cfg.stride, buffers.size());

    return cam;
}

extern "C" void pi_camera_destroy(pi_camera_t *cam) {
    if (!cam) return;
    teardown(cam);
    delete cam;
}

extern "C" int pi_camera_start(pi_camera_t *cam) {
    if (!cam || !cam->cam) return -1;
    if (cam->started.load()) return 0;

    cam->frame_counter.store(0, std::memory_order_relaxed);

    if (cam->cam->start() != 0) {
        PI_ERROR("camera", "Camera::start() failed");
        return -1;
    }
    cam->started = true;
    cam->running = true;

    // Prime the pump: queue every Request once so libcamera has a full
    // pipeline depth from frame zero. After this, requests cycle via
    // pi_camera_release.
    for (auto &req : cam->requests) {
        if (cam->cam->queueRequest(req.get()) != 0) {
            PI_ERROR("camera", "initial Camera::queueRequest() failed");
            cam->cam->stop();
            cam->started = false;
            cam->running = false;
            return -1;
        }
    }

    PI_INFO("camera", "started, %zu requests in flight", cam->requests.size());
    return 0;
}

extern "C" int pi_camera_stop(pi_camera_t *cam) {
    if (!cam || !cam->cam) return -1;
    if (!cam->started.load()) return 0;

    cam->running = false;
    if (cam->cam->stop() != 0) {
        PI_ERROR("camera", "Camera::stop() failed");
        return -1;
    }
    cam->started = false;

    // Drain any completions queued during the stop transition so a later
    // capture call doesn't return a stale frame.
    {
        std::lock_guard<std::mutex> lk(cam->completed_mtx);
        void *slot = nullptr;
        while (pi_ring_pop(cam->completed, &slot) == 0) {
            delete static_cast<CompletedSlot *>(slot);
        }
    }
    cam->completed_cv.notify_all();
    return 0;
}

extern "C" bool pi_camera_is_running(const pi_camera_t *cam) {
    return cam && cam->running.load();
}

extern "C" int pi_camera_capture(pi_camera_t       *cam,
                                 pi_camera_frame_t *out_frame,
                                 uint32_t           timeout_ms) {
    if (!cam || !out_frame) return -1;
    if (!cam->running.load()) return -1;

    void *slot_ptr = nullptr;

    // Fast path: a completed slot is already waiting.
    if (pi_ring_pop(cam->completed, &slot_ptr) != 0) {
        // Slow path: block on the condition variable until either a
        // completion arrives or the deadline elapses.
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lk(cam->completed_mtx);
        while (pi_ring_pop(cam->completed, &slot_ptr) != 0) {
            if (!cam->running.load()) return -1;
            if (cam->completed_cv.wait_until(lk, deadline) ==
                std::cv_status::timeout) {
                return -1;
            }
        }
    }

    auto *slot = static_cast<CompletedSlot *>(slot_ptr);
    if (!slot || !slot->mapped) return -1;

    const MappedBuffer *mb = slot->mapped;

    out_frame->pixels       = mb->y_plane;
    out_frame->pixels_size  = mb->y_size + mb->u_size + mb->v_size;
    out_frame->width        = cam->cfg.width;
    out_frame->height       = cam->cfg.height;
    out_frame->stride       = mb->y_stride; // back-compat alias for y_stride
    out_frame->timestamp_ns = slot->timestamp_ns;
    out_frame->frame_id     = slot->frame_id;
    out_frame->dma_fd       = mb->buffer->planes()[0].fd.get();
    out_frame->opaque       = slot;
    out_frame->y_stride     = mb->y_stride;
    out_frame->uv_stride    = mb->uv_stride;
    out_frame->u_offset     = static_cast<size_t>(mb->u_plane - mb->y_plane);
    out_frame->v_offset     = static_cast<size_t>(mb->v_plane - mb->y_plane);
    return 0;
}

extern "C" void pi_camera_release(pi_camera_t *cam, pi_camera_frame_t *frame) {
    if (!cam || !frame || !frame->opaque) return;

    auto *slot = static_cast<CompletedSlot *>(frame->opaque);
    libcamera::Request *req = slot->request;

    // Hand the buffer back to libcamera so this slot can be filled again.
    // ReuseBuffers preserves the addBuffer mapping we set up at create time.
    if (req) {
        req->reuse(libcamera::Request::ReuseBuffers);
        if (cam->started.load() && cam->cam) {
            if (cam->cam->queueRequest(req) != 0) {
                PI_WARN("camera", "queueRequest on release failed");
            }
        }
    }

    delete slot;

    // Zero the frame so accidental reuse traps loudly instead of silently
    // pointing at a recycled buffer.
    *frame = pi_camera_frame_t{};
    frame->dma_fd = -1;
}
