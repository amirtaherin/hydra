// Copyright (c) 2026 Amir Taherin
// Licensed under the MIT License (see LICENSE).
#pragma once
// Copyright (c) 2025 Amir Taherin
// Tegrastats logger for Hydra llama.cpp profiler.
// Base class: works on all Jetson platforms (Orin, Xavier, Thor).
// TegrastatsLoggerThor: adds NVML GPU/MEM utilization for Thor.
// Both produce nanosecond-precision timestamps matching the Python loggers.

#include <string>
#include <thread>
#include <atomic>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#include <dlfcn.h>
#include <cstring>

// ============================================================
// Base tegrastats logger — all platforms
// Runs "sudo tegrastats", reads stdout, prepends nanosecond
// timestamp, writes to file. Same format as Python tegrastats.py.
// ============================================================
class TegrastatsLogger {
public:
    TegrastatsLogger(const std::string& log_file, int interval_ms = 10)
        : log_file_(log_file), interval_ms_(interval_ms), running_(false), tgs_pid_(-1) {}

    virtual ~TegrastatsLogger() { stop(); }

    void start() {
        running_ = true;
        reader_thread_ = std::thread(&TegrastatsLogger::reader_loop, this);
        printf("Tegrastats started (C++ thread-based)\n");
    }

    void stop() {
        running_ = false;
        if (tgs_pid_ > 0) {
            kill(tgs_pid_, SIGTERM);
            waitpid(tgs_pid_, nullptr, 0);
            tgs_pid_ = -1;
        }
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
        printf("Tegrastats stopped successfully\n");
    }

protected:
    // Override this in subclass to append extra data per line
    virtual std::string extra_fields() { return ""; }

private:
    std::string log_file_;
    int interval_ms_;
    std::atomic<bool> running_;
    pid_t tgs_pid_;
    std::thread reader_thread_;

    // Get current time as "MM-DD-YYYY HH:MM:SS.NNNNNNNNN"
    static std::string nanosecond_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
        time_t sec = epoch_ns / 1000000000LL;
        long ns = epoch_ns % 1000000000LL;

        struct tm tm_local;
        localtime_r(&sec, &tm_local);

        char buf[64];
        snprintf(buf, sizeof(buf), "%02d-%02d-%04d %02d:%02d:%02d.%09ld",
                 tm_local.tm_mon + 1, tm_local.tm_mday, tm_local.tm_year + 1900,
                 tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec, ns);
        return std::string(buf);
    }

    void reader_loop() {
        // Create pipe and fork tegrastats
        int pipefd[2];
        if (pipe(pipefd) != 0) {
            fprintf(stderr, "Error: pipe() failed for tegrastats\n");
            return;
        }

        tgs_pid_ = fork();
        if (tgs_pid_ == 0) {
            // Child: redirect stdout to pipe, exec tegrastats
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
            std::string interval_str = std::to_string(interval_ms_);
            execl("/usr/bin/sudo", "sudo", "/usr/bin/tegrastats",
                  "--interval", interval_str.c_str(), nullptr);
            _exit(1);
        }

        // Parent: read from pipe
        close(pipefd[1]);
        FILE* pipe_fp = fdopen(pipefd[0], "r");
        if (!pipe_fp) {
            fprintf(stderr, "Error: fdopen() failed for tegrastats pipe\n");
            return;
        }

        std::ofstream outfile(log_file_);
        char line[4096];

        while (running_ && fgets(line, sizeof(line), pipe_fp)) {
            // Strip trailing newline
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

            std::string ts = nanosecond_timestamp();
            std::string extra = extra_fields();

            outfile << ts << " " << line;
            if (!extra.empty()) outfile << " " << extra;
            outfile << "\n";
            outfile.flush();
        }

        fclose(pipe_fp);
        outfile.close();
    }
};


// ============================================================
// Thor tegrastats logger — adds NVML GPU_UTIL and MEM_UTIL
// Uses dlopen to load libnvidia-ml.so.1 at runtime.
// ============================================================
class TegrastatsLoggerThor : public TegrastatsLogger {
public:
    TegrastatsLoggerThor(const std::string& log_file, int interval_ms = 10)
        : TegrastatsLogger(log_file, interval_ms), nvml_handle_(nullptr),
          nvml_init_(nullptr), nvml_shutdown_(nullptr),
          nvml_get_handle_(nullptr), nvml_get_util_(nullptr) {
        init_nvml();
    }

    ~TegrastatsLoggerThor() override {
        shutdown_nvml();
    }

protected:
    std::string extra_fields() override {
        unsigned int gpu_util = 0, mem_util = 0;
        if (nvml_get_util_ && device_handle_) {
            // nvmlUtilization_t is two unsigned ints: {gpu, memory}
            struct { unsigned int gpu; unsigned int memory; } util = {0, 0};
            nvml_get_util_(device_handle_, &util);
            gpu_util = util.gpu;
            mem_util = util.memory;
        }
        return "GPU_UTIL " + std::to_string(gpu_util) + "% MEM_UTIL " + std::to_string(mem_util) + "%";
    }

private:
    void* nvml_handle_;   // dlopen handle
    void* device_handle_; // nvmlDevice_t

    // Function pointers
    using nvmlInit_t = int (*)();
    using nvmlShutdown_t = int (*)();
    using nvmlGetHandle_t = int (*)(unsigned int, void**);
    using nvmlGetUtil_t = int (*)(void*, void*);

    nvmlInit_t nvml_init_;
    nvmlShutdown_t nvml_shutdown_;
    nvmlGetHandle_t nvml_get_handle_;
    nvmlGetUtil_t nvml_get_util_;

    void init_nvml() {
        nvml_handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!nvml_handle_) {
            fprintf(stderr, "Warning: Could not load libnvidia-ml.so.1: %s\n", dlerror());
            return;
        }

        nvml_init_ = (nvmlInit_t)dlsym(nvml_handle_, "nvmlInit_v2");
        nvml_shutdown_ = (nvmlShutdown_t)dlsym(nvml_handle_, "nvmlShutdown");
        nvml_get_handle_ = (nvmlGetHandle_t)dlsym(nvml_handle_, "nvmlDeviceGetHandleByIndex_v2");
        nvml_get_util_ = (nvmlGetUtil_t)dlsym(nvml_handle_, "nvmlDeviceGetUtilizationRates");

        if (!nvml_init_ || !nvml_shutdown_ || !nvml_get_handle_ || !nvml_get_util_) {
            fprintf(stderr, "Warning: NVML symbol lookup failed\n");
            dlclose(nvml_handle_);
            nvml_handle_ = nullptr;
            return;
        }

        nvml_init_();
        device_handle_ = nullptr;
        nvml_get_handle_(0, &device_handle_);
    }

    void shutdown_nvml() {
        if (nvml_shutdown_) nvml_shutdown_();
        if (nvml_handle_) dlclose(nvml_handle_);
        nvml_handle_ = nullptr;
    }
};


// ============================================================
// Factory: create the right logger based on hostname
// ============================================================
inline TegrastatsLogger* create_tegrastats_logger(const std::string& log_file, int interval_ms = 10) {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        std::string h(hostname);
        if (h.find("thor") != std::string::npos || h.find("Thor") != std::string::npos) {
            return new TegrastatsLoggerThor(log_file, interval_ms);
        }
    }
    return new TegrastatsLogger(log_file, interval_ms);
}
