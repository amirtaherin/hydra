// Copyright (c) 2026 Amir Taherin
// Licensed under the MIT License (see LICENSE).
#include "llama.cpp/include/llama-cpp.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <map>
#include <chrono>
#include <numeric>
#include "nlohmann/json.hpp"
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>
#include <sstream>
#include <array>
#include <regex>
#include "tegrastats_logger.h"
#include <cuda_runtime.h>

#define DEFAULT_MODELS_JSON "../models/models.json"
#define DEFAULT_INPUT_DATA "../dataset/input_data.jsonl"
#define DEFAULT_RESULTS_DIR "../results/"
#define DEFAULT_CACHE_DIR "../GGUF-Models/"  // local cache for downloaded GGUF files

using json = nlohmann::json;


using high_res_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<high_res_clock>;

double to_seconds(const time_point &start, const time_point &end) {
    return std::chrono::duration<double>(end - start).count();
}

int64_t to_nanoseconds(const time_point& tp) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

// Wall-clock nanoseconds for CSV start_time/end_time (must match tegrastats timestamps)
int64_t wall_clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double calculate_mean(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculate_variance(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double mean = calculate_mean(values);
    double variance = 0.0;
    for (const auto& val : values) {
        variance += (val - mean) * (val - mean);
    }
    return variance / values.size();
}

// Run a shell command and capture stdout
std::string exec_cmd(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

// Resolve HF repo + dtype tag to actual GGUF filename and expected size via HuggingFace manifest API
// e.g., repo="EShawC/Llama-3.1-8B-GGUF", dtype="Q4_K_M" -> {"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 1234567}
struct GgufShard {
    std::string filename;
    uintmax_t expected_size = 0;
};

struct GgufFileInfo {
    std::vector<GgufShard> shards;
};

GgufFileInfo resolve_gguf_file_info(const std::string& repo, const std::string& dtype) {
    std::string cmd = "curl -s -H 'User-Agent: llama-cpp' "
                      "'https://huggingface.co/v2/" + repo + "/manifests/" + dtype + "' 2>/dev/null";
    std::string response = exec_cmd(cmd);
    if (response.empty()) {
        fprintf(stderr, "Error: Failed to query HF manifest for %s:%s\n", repo.c_str(), dtype.c_str());
        return {};
    }

    try {
        auto manifest = json::parse(response);

        // Split GGUF: multiple shards under "ggufFiles"
        if (manifest.contains("ggufFiles") && manifest["ggufFiles"].is_array()) {
            GgufFileInfo info;
            for (const auto& entry : manifest["ggufFiles"]) {
                if (!entry.contains("rfilename")) continue;
                GgufShard shard;
                shard.filename = entry["rfilename"].get<std::string>();
                if (entry.contains("size")) {
                    shard.expected_size = entry["size"].get<uintmax_t>();
                }
                info.shards.push_back(shard);
            }
            if (!info.shards.empty()) return info;
        }

        // Single (or first shard) under "ggufFile"
        if (manifest.contains("ggufFile") && manifest["ggufFile"].contains("rfilename")) {
            GgufFileInfo info;
            GgufShard shard;
            shard.filename = manifest["ggufFile"]["rfilename"].get<std::string>();
            if (manifest["ggufFile"].contains("size")) {
                shard.expected_size = manifest["ggufFile"]["size"].get<uintmax_t>();
            }
            info.shards.push_back(shard);

            // Detect split pattern e.g. "-00001-of-00003.gguf" and generate remaining shards
            std::regex split_re(R"((-(\d{5})-of-(\d{5}))\.gguf$)");
            std::smatch match;
            if (std::regex_search(shard.filename, match, split_re)) {
                int total = std::stoi(match[3].str());
                int width = (int)match[2].str().size();
                std::string prefix = shard.filename.substr(0, match.position(1));
                std::string total_str = match[3].str();
                for (int k = 2; k <= total; k++) {
                    std::ostringstream oss;
                    oss << prefix << "-" << std::setfill('0') << std::setw(width) << k
                        << "-of-" << total_str << ".gguf";
                    GgufShard extra;
                    extra.filename = oss.str();
                    info.shards.push_back(extra);
                }
                printf("Split GGUF detected: %d shards\n", total);
            }

            return info;
        }

        fprintf(stderr, "Error: No ggufFile(s) in manifest for %s:%s\n", repo.c_str(), dtype.c_str());
        return {};
    } catch (const json::parse_error& e) {
        fprintf(stderr, "Error: Failed to parse HF manifest JSON: %s\n", e.what());
        return {};
    }
}

// Download a GGUF model from HuggingFace if not already cached locally
// Returns the local file path, or empty string on failure
std::string download_gguf_model(const std::string& repo, const std::string& dtype,
                                const std::string& cache_dir) {
    // 1. Resolve the actual filename(s) and expected size(s) from HF manifest API
    printf("Resolving %s:%s from HuggingFace...\n", repo.c_str(), dtype.c_str());
    GgufFileInfo info = resolve_gguf_file_info(repo, dtype);
    if (info.shards.empty()) return "";

    std::filesystem::create_directories(cache_dir);
    std::string first_shard_path;

    for (size_t s = 0; s < info.shards.size(); s++) {
        const auto& shard = info.shards[s];
        std::string local_path = cache_dir + shard.filename;
        if (s == 0) first_shard_path = local_path;

        // 2. Check if file exists locally in cache with correct size
        if (std::filesystem::exists(local_path)) {
            uintmax_t actual_size = std::filesystem::file_size(local_path);
            if (shard.expected_size > 0 && actual_size != shard.expected_size) {
                printf("Cached shard is incomplete (%ju / %ju bytes), re-downloading...\n",
                       actual_size, shard.expected_size);
                std::filesystem::remove(local_path);
            } else {
                printf("Found cached: %s\n", local_path.c_str());
                continue;
            }
        }

        // 3. Download from HuggingFace
        std::string url = "https://huggingface.co/" + repo + "/resolve/main/" + shard.filename;
        printf("Downloading [%zu/%zu] %s -> %s\n", s + 1, info.shards.size(),
               url.c_str(), local_path.c_str());

        std::string cmd = "curl -L --progress-bar -o '" + local_path + "' '" + url + "'";
        int ret = system(cmd.c_str());
        if (ret != 0 || !std::filesystem::exists(local_path)) {
            fprintf(stderr, "Error: Download failed for %s\n", url.c_str());
            return "";
        }

        // Verify downloaded file size
        if (shard.expected_size > 0) {
            uintmax_t actual_size = std::filesystem::file_size(local_path);
            if (actual_size != shard.expected_size) {
                fprintf(stderr, "Error: Downloaded file size mismatch (%ju != %ju bytes)\n",
                        actual_size, shard.expected_size);
                std::filesystem::remove(local_path);
                return "";
            }
        }

        printf("Downloaded: %s\n", local_path.c_str());
    }

    return first_shard_path;
}

// Model entry from models.json
struct ModelEntry {
    std::string name;
    std::string hf_path;
    std::string gguf_path;
};

// Read model list from models.json
std::vector<ModelEntry> collect_models_from_json(const std::string& json_path) {
    std::vector<ModelEntry> models;
    std::ifstream f(json_path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: Could not open models file: %s\n", json_path.c_str());
        return models;
    }

    auto data = json::parse(f);
    for (const auto& m : data["models"]) {
        ModelEntry entry;
        entry.name = m["name"].get<std::string>();
        entry.hf_path = m.value("hf_path", "");
        entry.gguf_path = m.value("gguf_path", "");
        models.push_back(entry);
    }
    return models;
}

// grabs all prompts from the input data file and stores it in a map which has the key and the prompt as the value
std::map<int, std::string> collect_prompts(std::string input_dataset_path, int &n_prompts) {
    std::map<int, std::string> prompts;
    std::ifstream inputFile(input_dataset_path);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open " << input_dataset_path << std::endl;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        auto jsonl_data = json::parse(line);
        std::string prompt = jsonl_data["prompt"];
        prompts[n_prompts] = prompt;
        n_prompts++;
    }

    return prompts;
}

// CLI flag parsers
std::string get_flag_string(int argc, char* argv[], const std::string& flag, const std::string& default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == flag) return argv[i + 1];
    }
    return default_val;
}

int get_flag_int(int argc, char* argv[], const std::string& flag, int default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == flag) return std::atoi(argv[i + 1]);
    }
    return default_val;
}

// Determine output directory from --result-dir flag or hostname
std::string get_results_dir(int argc, char* argv[]) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--result-dir") {
            std::string dir = argv[i + 1];
            if (dir.back() != '/') dir += '/';
            return dir;
        }
    }
    // Derive from hostname to match Hydra convention (results_orin/, results_xavier/, etc.)
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        std::string h(hostname);
        std::string platform;
        if (h.find("orin") != std::string::npos || h.find("Orin") != std::string::npos) {
            platform = "orin";
        } else if (h.find("xavier") != std::string::npos || h.find("Xavier") != std::string::npos) {
            platform = "xavier";
        } else if (h.find("thor") != std::string::npos || h.find("Thor") != std::string::npos) {
            platform = "thor";
        } else {
            platform = h;
        }
        return "../results_" + platform + "/";
    }
    return DEFAULT_RESULTS_DIR;
}

int main(int argc, char* argv[]) {
    // 1. loads all compute backends
    ggml_backend_load_all();

    std::string results_dir = get_results_dir(argc, argv);
    std::string input_path = get_flag_string(argc, argv, "--input", DEFAULT_INPUT_DATA);
    std::string model_names_path = get_flag_string(argc, argv, "--model-names", DEFAULT_MODELS_JSON);
    std::string dtype = get_flag_string(argc, argv, "--dtype", "Q4_K_M");
    std::string cache_dir = DEFAULT_CACHE_DIR;
    int max_prompts_limit = get_flag_int(argc, argv, "--max-prompts", -1);
    int max_new_tokens = get_flag_int(argc, argv, "--max-new-tokens", 500);
    // Context/batch size. Default 2048 matches the main-corpus collection
    // configuration and fits Xavier's NVMAP allocator limits even for 7-8B
    // models. The sensitivity sweeps (S1 inputs ~5K, S2 outputs 5K) need
    // 8192 - their runner scripts pass --n-ctx 8192 explicitly (Orin/Thor
    // only; large n_ubatch inflates the compute buffer by the vocab-sized
    // logits tensor, which exceeds Xavier's allocator ceiling on 7B models).
    int n_ctx_arg = get_flag_int(argc, argv, "--n-ctx", 2048);
    bool download_only = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--download-only") { download_only = true; break; }
    }

    if (cache_dir.back() != '/') cache_dir += '/';
    std::filesystem::create_directories(results_dir);
    std::filesystem::create_directories(cache_dir);

    printf("Results directory: %s\n", results_dir.c_str());
    printf("Input data: %s\n", input_path.c_str());
    printf("Model names: %s\n", model_names_path.c_str());
    printf("Dtype: %s\n", dtype.c_str());
    printf("Max new tokens: %d\n", max_new_tokens);

    // 2. collect model entries
    std::vector<ModelEntry> model_entries = collect_models_from_json(model_names_path);
    if (model_entries.empty()) {
        std::cout << "Something went wrong with collecting the models. "
                  << "Double check the file locations match up in your file system and script." << std::endl;
        return 0;
    }

    // Download-only mode: resolve and download all GGUFs, then exit
    if (download_only) {
        printf("\n--- Download-only mode ---\n");
        int success = 0, failed = 0;
        for (size_t i = 0; i < model_entries.size(); i++) {
            const auto& entry = model_entries[i];
            std::string model_path = download_gguf_model(entry.gguf_path, dtype, cache_dir);
            if (model_path.empty()) {
                fprintf(stderr, "Failed: %s\n", entry.name.c_str());
                failed++;
            } else {
                success++;
            }
        }
        printf("\nDownload complete: %d succeeded, %d failed\n", success, failed);
        return (failed > 0) ? 1 : 0;
    }

    // Collect prompts (only needed for inference, not download-only)
    int n_prompts = 0;
    std::map<int, std::string> prompts = collect_prompts(input_path, n_prompts);
    if (n_prompts == 0) {
        std::cout << "Something went wrong with collecting the prompts. "
                  << "Double check the file locations match up in your file system and script." << std::endl;
        return 0;
    }

    int prompts_to_run = n_prompts;
    if (max_prompts_limit > 0 && max_prompts_limit < n_prompts) {
        prompts_to_run = max_prompts_limit;
    }

    // 3. parameters for model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    // Context (runtime settings)
    llama_context_params ctx_params = llama_context_default_params();
    // Context window: total input + output tokens (see --n-ctx above).
    ctx_params.n_ctx = n_ctx_arg;
    // Batch sizes must be >= the largest prefill in one llama_decode call,
    // or it fails GGML_ASSERT(n_tokens_all <= n_batch). Match n_ctx.
    ctx_params.n_batch  = n_ctx_arg;
    ctx_params.n_ubatch = n_ctx_arg;
    // if threads equal the number of cores, you are maxing out load for the CPU, since we are using GPU this doesn't matter much
    ctx_params.n_threads = 8;
    // Enable flash attention on SM 8.0+ (Orin=8.7, Thor=11.0); Xavier (7.2) does not support it
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        if (props.major > 8 || (props.major == 8 && props.minor >= 0)) {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            printf("Flash attention: ENABLED (SM %d.%d)\n", props.major, props.minor);
        } else {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            printf("Flash attention: DISABLED (SM %d.%d)\n", props.major, props.minor);
        }
    }
    // enable internal timing (defaults to true/disabled)
    ctx_params.no_perf = false;


    for (size_t i = 0; i < model_entries.size(); i++) {
        const auto& entry = model_entries[i];

        // 4. Resolve and download GGUF model from HuggingFace
        std::string model_path = download_gguf_model(entry.gguf_path, dtype, cache_dir);
        if (model_path.empty()) {
            fprintf(stderr, "Skipping model %s: failed to resolve/download GGUF\n", entry.name.c_str());
            continue;
        }

        // 5. load model from file using api
        llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) {
                fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
                continue;
        }
        // 6. create context using parameters from above
        llama_context* ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
                fprintf(stderr, "Failed to create context\n");
                llama_model_free(model);
                continue;
        }
        std::string model_name = entry.name;
        size_t model_size = llama_model_size(model);

        std::string tgs_log = results_dir + "TGS_" + model_name + "_" + dtype + "_.csv";
        TegrastatsLogger* tgs = create_tegrastats_logger(tgs_log, 10);
        tgs->start();

        std::ofstream csvfile(results_dir + "INFO_" + model_name + "_" + dtype + "_.csv");

        csvfile << "prompt_number,start_time,end_time,model_name,size_of_model,"
                << "tokenization_time,prefill_time,time_to_first_token,"
                << "mean_token_generation_time,variance_token_generation_time,"
                << "mean_token_decode_time,variance_token_decode_time,"
                << "mean_inter_token_latency,variance_inter_token_latency,"
                << "end_to_end_latency,estimated_end_to_end_latency,"
                << "output_length,total_generation_time,"
                << "total_number_of_prompts,total_number_of_tokens,"
                << "total_number_of_tokens_in_input,total_number_of_tokens_in_output,"
                << "total_number_of_tokens_in_input_plus_output,"
                << "prompt_tokens_per_sec,generation_tokens_per_sec,"
                << "llama_t_p_eval_ms,llama_t_eval_ms,llama_n_p_eval,llama_n_eval\n";
        int counter = 0;


        for (int j = 0; j < prompts_to_run; j++) {
                counter++;
                printf("\nProcessing prompt %d/%d...\n", counter, prompts_to_run);
                int64_t START = wall_clock_ns();
                // clear the kv cache between prompts
                llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
                // reset llama.cpp internal perf counters for cross-validation
                llama_perf_context_reset(ctx);

                // 6. Tokenization
                const char *prompt = prompts[j].c_str();
                const llama_vocab* vocab = llama_model_get_vocab(model);
                auto t0 = high_res_clock::now();
                int n_tokens = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);

                // Bug 5: Replace VLA with std::vector to avoid stack overflow on long prompts
                std::vector<llama_token> tokens(n_tokens);
                llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), n_tokens, true, true);
                auto t1 = high_res_clock::now();
                double tokenization_time = to_seconds(t0, t1);

                // 7. Create batch and add tokens
                // Bug 5: Ensure batch capacity is at least n_tokens
                int batch_capacity = std::max(512, n_tokens);
                llama_batch batch = llama_batch_init(batch_capacity, 0, 1);

                for (int k = 0; k < n_tokens; k++) {
                    batch.token[k]    = tokens[k];
                    batch.pos[k]      = k;
                    batch.n_seq_id[k] = 1;
                    batch.seq_id[k][0] = 0;
                    batch.logits[k]   = (k == n_tokens - 1);
                }
                batch.n_tokens = n_tokens;

                // 8. Prefill: decode the entire prompt at once
                // Bug 3: t2 is now immediately before llama_decode, not before the batch population loop
                // Bug 2: llama_synchronize() before timing endpoints for accurate GPU timing
                llama_synchronize(ctx);
                auto t2 = high_res_clock::now();
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "Failed to decode (prefill)\n");
                    llama_batch_free(batch);
                    continue;
                }
                llama_synchronize(ctx);
                auto t3 = high_res_clock::now();
                double prefill_time = to_seconds(t2, t3);

                // 9. Token generation loop with corrected timing
                // Bug 1 fix: llama_decode() is now INSIDE the timing bracket for each generated token.
                // For the first token, logits are already available from prefill (no decode needed).
                // For subsequent tokens, llama_decode() runs first, then logits are read.
                // printf("Response: ");
                int n_cur = n_tokens;
                int n_max = n_tokens + max_new_tokens;
                llama_token new_token_id = 0;
                std::vector<double> token_generation_times;
                std::vector<double> token_decode_times;
                std::vector<time_point> generation_start_times;
                std::vector<time_point> generation_end_times;
                time_point last_decode_end;
                bool first_token = true;


                while (n_cur < n_max) {
                    // Bug 2: synchronize before starting the timer to ensure previous GPU work is done
                    llama_synchronize(ctx);
                    auto t_start = high_res_clock::now();

                    // Bug 1: llama_decode() is now inside the timing bracket.
                    // Skip decode on the first iteration — logits are from the prefill.
                    if (!first_token) {
                        if (llama_decode(ctx, batch) != 0) {
                            fprintf(stderr, "\nFailed to decode\n");
                            break;
                        }
                        // Bug 2: synchronize after decode to ensure GPU results are ready
                        llama_synchronize(ctx);
                    }

                    float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
                    int32_t n_vocab = llama_vocab_n_tokens(vocab);

                    new_token_id = 0;
                    float max_logit = logits[0];
                    for (int v = 1; v < n_vocab; v++) {
                        if (logits[v] > max_logit) {
                            max_logit = logits[v];
                            new_token_id = v;
                        }
                    }
                    auto t_end = high_res_clock::now();

                    // No EOG early stop — always generate max_new_tokens to match
                    // the HuggingFace profiler (llm_profiler/llm_inference.py:164)

                    // convert token to text (token decode timing, separate from generation)
                    char buf[128];
                    auto dec_start = high_res_clock::now();
                    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                    auto dec_end = high_res_clock::now();

                    double token_time = to_seconds(t_start, t_end);
                    double decode_time = to_seconds(dec_start, dec_end);
                    token_generation_times.push_back(token_time);
                    token_decode_times.push_back(decode_time);
                    generation_start_times.push_back(t_start);
                    generation_end_times.push_back(t_end);
                    last_decode_end = dec_end;
                    // if (n > 0) {
                    //     printf("%.*s", n, buf);
                    //     fflush(stdout);
                    // }

                    // prepare batch for the next decode call
                    batch.n_tokens = 1;
                    batch.token[0] = new_token_id;
                    batch.pos[0] = n_cur;
                    batch.n_seq_id[0] = 1;
                    batch.seq_id[0][0] = 0;
                    batch.logits[0] = true;
                    n_cur++;
                    first_token = false;
                }
                auto end_hrc = high_res_clock::now();  // for e2e duration
                int64_t END = wall_clock_ns();         // wall-clock for CSV/TGS matching
                // printf("\n");

                // Bug 4: Guard against empty output — if first token is EOG,
                // generation_end_times is empty and .size()-1 wraps to SIZE_MAX.
                std::vector<double> inter_token_latencies;
                if (generation_end_times.size() > 1) {
                    for (size_t k = 0; k < generation_end_times.size() - 1; k++) {
                        inter_token_latencies.push_back(to_seconds(generation_end_times[k], generation_end_times[k + 1]));
                    }
                }

                int output_length = static_cast<int>(token_generation_times.size());
                int total_tokens = n_tokens + output_length;

                // Bug 4: Guard access to first element for TTFT
                double time_to_first_token = tokenization_time + prefill_time;
                if (!token_generation_times.empty()) {
                    time_to_first_token += token_generation_times[0] + token_decode_times[0];
                }

                double total_generation_time = 0.0;
                if (!generation_start_times.empty()) {
                    total_generation_time = to_seconds(generation_start_times.front(), last_decode_end);
                }

                double e2e_latency = to_seconds(t0, end_hrc);
                double estimated_e2e_latency = tokenization_time + prefill_time + total_generation_time;

                // tokens/sec computation
                double prompt_tokens_per_sec = (prefill_time > 0.0) ? n_tokens / prefill_time : 0.0;
                double generation_tokens_per_sec = (total_generation_time > 0.0) ? output_length / total_generation_time : 0.0;

                // llama_perf_context() cross-validation
                struct llama_perf_context_data perf = llama_perf_context(ctx);
                // printf("[Cross-validation] llama.cpp internal: prefill=%.2fms (%d tokens, %.1f t/s), "
                //        "eval=%.2fms (%d tokens, %.1f t/s)\n",
                //        perf.t_p_eval_ms, (int)perf.n_p_eval,
                //        (perf.t_p_eval_ms > 0) ? perf.n_p_eval / (perf.t_p_eval_ms / 1000.0) : 0.0,
                //        perf.t_eval_ms, (int)perf.n_eval,
                //        (perf.t_eval_ms > 0) ? perf.n_eval / (perf.t_eval_ms / 1000.0) : 0.0);
                // printf("[Cross-validation] Manual timing:      prefill=%.2fms (%d tokens, %.1f t/s), "
                //        "eval=%.2fms (%d tokens, %.1f t/s)\n",
                //        prefill_time * 1000.0, n_tokens, prompt_tokens_per_sec,
                //        total_generation_time * 1000.0, output_length, generation_tokens_per_sec);

                csvfile << std::fixed << std::setprecision(17);
                csvfile << counter << ","
                        << START << ","
                        << END << ","
                        << model_name << ","
                        << model_size << ","
                        << tokenization_time << ","
                        << prefill_time << ","
                        << time_to_first_token << ","
                        << calculate_mean(token_generation_times) << ","
                        << calculate_variance(token_generation_times) << ","
                        << calculate_mean(token_decode_times) << ","
                        << calculate_variance(token_decode_times) << ","
                        << calculate_mean(inter_token_latencies) << ","
                        << calculate_variance(inter_token_latencies) << ","
                        << e2e_latency << ","
                        << estimated_e2e_latency << ","
                        << output_length << ","
                        << total_generation_time << ","
                        << n_prompts << ","
                        << total_tokens << ","
                        << n_tokens << ","
                        << output_length << ","
                        << total_tokens << ","
                        << prompt_tokens_per_sec << ","
                        << generation_tokens_per_sec << ","
                        << perf.t_p_eval_ms << ","
                        << perf.t_eval_ms << ","
                        << perf.n_p_eval << ","
                        << perf.n_eval << "\n";
                csvfile.flush();
                llama_batch_free(batch);
        }
        csvfile.close();
        tgs->stop();
        delete tgs;
        llama_free(ctx);
        llama_model_free(model);
    }
    printf("\nDone.\n");
    llama_backend_free();
    return 0;
}
