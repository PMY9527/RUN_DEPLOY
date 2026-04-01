// Shared memory structure for real-time CMG visualization.
// Writer: State_RLResidual policy thread
// Reader: CmgVizThread in MuJoCo viewer (or visualize_cmg.py fallback)
#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <iostream>

#define CMG_VIZ_SHM_NAME "/cmg_viz_data"
#define CMG_VIZ_NUM_JOINTS 29
#define CMG_VIZ_CMD_DIM 3

struct CMGVizData {
    std::atomic<uint32_t> seq;          // write sequence (reader uses to detect updates)
    uint64_t timestamp_us;              // microsecond timestamp since epoch
    float qref[CMG_VIZ_NUM_JOINTS];    // CMG reference joint positions (USD order)
    float qref_vel[CMG_VIZ_NUM_JOINTS]; // CMG reference joint velocities (USD order)
    float actual_pos[CMG_VIZ_NUM_JOINTS]; // actual joint positions (USD order)
    float actual_vel[CMG_VIZ_NUM_JOINTS]; // actual joint velocities (USD order)
    float command[CMG_VIZ_CMD_DIM];     // velocity commands [vx, vy, yaw_rate]
    float raw_residual[CMG_VIZ_NUM_JOINTS]; // policy residual output (USD order)
    float combined[CMG_VIZ_NUM_JOINTS]; // final action: qref + residual (USD order)
    float ctrl[CMG_VIZ_NUM_JOINTS];     // PD ctrl torques: kp*(action-pos) - kd*vel (USD order)
};

class CMGVizWriter {
public:
    CMGVizWriter() {
        int fd = shm_open(CMG_VIZ_SHM_NAME, O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            std::cerr << "[CMGViz] shm_open failed\n";
            return;
        }
        if (ftruncate(fd, sizeof(CMGVizData)) < 0) {
            std::cerr << "[CMGViz] ftruncate failed\n";
            close(fd);
            return;
        }
        data_ = static_cast<CMGVizData*>(
            mmap(nullptr, sizeof(CMGVizData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        close(fd);
        if (data_ == MAP_FAILED) {
            std::cerr << "[CMGViz] mmap failed\n";
            data_ = nullptr;
            return;
        }
        std::memset(data_, 0, sizeof(CMGVizData));
        std::cout << "[CMGViz] Shared memory writer initialized\n";
    }

    ~CMGVizWriter() {
        if (data_) {
            munmap(data_, sizeof(CMGVizData));
        }
    }

    void write(const std::vector<float>& qref,
               const std::vector<float>& qref_vel,
               const std::vector<float>& actual_pos,
               const std::vector<float>& actual_vel,
               const std::vector<float>& command,
               const std::vector<float>& raw_residual,
               const std::vector<float>& combined,
               const std::vector<float>& ctrl = {}) {
        if (!data_) return;

        auto now = std::chrono::high_resolution_clock::now();
        data_->timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        auto copy = [](float* dst, const std::vector<float>& src, size_t n) {
            size_t count = std::min(src.size(), n);
            std::memcpy(dst, src.data(), count * sizeof(float));
        };

        copy(data_->qref, qref, CMG_VIZ_NUM_JOINTS);
        copy(data_->qref_vel, qref_vel, CMG_VIZ_NUM_JOINTS);
        copy(data_->actual_pos, actual_pos, CMG_VIZ_NUM_JOINTS);
        copy(data_->actual_vel, actual_vel, CMG_VIZ_NUM_JOINTS);
        copy(data_->command, command, CMG_VIZ_CMD_DIM);
        copy(data_->raw_residual, raw_residual, CMG_VIZ_NUM_JOINTS);
        copy(data_->combined, combined, CMG_VIZ_NUM_JOINTS);
        copy(data_->ctrl, ctrl, CMG_VIZ_NUM_JOINTS);

        data_->seq.fetch_add(1, std::memory_order_release);
    }

    bool ok() const { return data_ != nullptr; }

private:
    CMGVizData* data_ = nullptr;
};

class CMGVizReader {
public:
    CMGVizReader() = default;

    // Returns true if new data was read (sequence number changed).
    bool read(CMGVizData& out) {
        if (!tryConnect()) return false;
        uint32_t cur = data_->seq.load(std::memory_order_acquire);
        if (cur == last_seq_) return false;
        // Copy the data snapshot
        std::memcpy(&out, data_, sizeof(CMGVizData));
        last_seq_ = cur;
        return true;
    }

    bool connected() const { return data_ != nullptr; }

private:
    bool tryConnect() {
        if (data_) return true;
        int fd = shm_open(CMG_VIZ_SHM_NAME, O_RDONLY, 0);
        if (fd < 0) return false;
        void* ptr = mmap(nullptr, sizeof(CMGVizData), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) return false;
        data_ = static_cast<CMGVizData*>(ptr);
        std::cout << "[CMGViz] Shared memory reader connected\n";
        return true;
    }

    CMGVizData* data_ = nullptr;
    uint32_t last_seq_ = 0;
};
