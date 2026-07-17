#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "onnxruntime_cxx_api.h"
#include "pffft.h"

namespace py = pybind11;

inline float clip_sample(float x) {
    if (std::isnan(x) || std::isinf(x)) return 0.0f;
    if (x > 1.0f) return 1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

void clip_buffer(float* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        data[i] = clip_sample(data[i]);
    }
}

static const size_t FRAME_SIZE = 960;
static const size_t HOP_LENGTH = 480;
static const float SAMPLE_RATE = 48000.0f;

inline std::vector<float> create_hann_window(int nfft = 960) {
    std::vector<float> w(nfft);
    for (int i = 0; i < nfft; ++i) {
        float hann = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / nfft);
        w[i] = std::sqrt(hann + 1e-10f);
    }
    return w;
}

// ============================================================================
// AEC (Acoustic Echo Cancellation) — aec7 ONNX streaming
// ============================================================================

class AecProcessor {
public:
    static const int AEC_FS = 48000;
    static const int AEC_WIN = 960;
    static const int AEC_HOP = 480;
    static const int AEC_NFFT = 960;
    static const int AEC_FREQ = 481;

    // aec7 cache sizes
    static constexpr int RES_ENC_CONV_SIZE  = 135680;
    static constexpr int RES_ENC_TFA_SIZE   = 248;
    static constexpr int MIC_ENC_CONV_SIZE  = 135680;
    static constexpr int MIC_ENC_TFA_SIZE   = 248;
    static constexpr int DEEP_ENC_TFA_SIZE  = 336;
    static constexpr int DEC_CONV_SIZE      = 13440;
    static constexpr int DEC_TFA_SIZE       = 496;
    static constexpr int INTER_SIZE         = 7680;
    static constexpr int PREV_SIZE          = 320;

    static const int NUM_INPUTS = 14;
    static const int NUM_OUTPUTS = 14;
    static const int DEEP_ENC_CONV_OUTPUT_IDX = 5;

    AecProcessor(const std::string& model_path)
        : mic_buffer_(AEC_WIN, 0.0f),
          far_buffer_(AEC_WIN, 0.0f),
          ola_accumulator_(AEC_NFFT, 0.0f),
          window_sum_(AEC_NFFT, 0.0f)
    {
        window_.resize(AEC_NFFT);
        for (int i = 0; i < AEC_NFFT; ++i) {
            float hann = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (AEC_NFFT - 1)));
            window_[i] = std::sqrt(hann);
        }

        res_enc_conv_.resize(RES_ENC_CONV_SIZE, 0.0f);
        res_enc_tfa_.resize(RES_ENC_TFA_SIZE, 0.0f);
        mic_enc_conv_.resize(MIC_ENC_CONV_SIZE, 0.0f);
        mic_enc_tfa_.resize(MIC_ENC_TFA_SIZE, 0.0f);
        deep_enc_tfa_.resize(DEEP_ENC_TFA_SIZE, 0.0f);
        dec_conv_.resize(DEC_CONV_SIZE, 0.0f);
        dec_tfa_.resize(DEC_TFA_SIZE, 0.0f);
        inter_.resize(INTER_SIZE, 0.0f);
        res_prev1_.resize(PREV_SIZE, 0.0f);
        res_prev2_.resize(PREV_SIZE, 0.0f);
        mic_prev1_.resize(PREV_SIZE, 0.0f);
        mic_prev2_.resize(PREV_SIZE, 0.0f);

        fft_in_  = static_cast<float*>(pffft_aligned_malloc(AEC_NFFT * sizeof(float)));
        fft_out_ = static_cast<float*>(pffft_aligned_malloc(AEC_NFFT * sizeof(float)));
        ifft_out_= static_cast<float*>(pffft_aligned_malloc(AEC_NFFT * sizeof(float)));
        mic_onnx_.resize(AEC_FREQ * 2, 0.0f);
        far_onnx_.resize(AEC_FREQ * 2, 0.0f);

        fft_plan_ = pffft_new_setup(AEC_NFFT, PFFFT_REAL);

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ORT_SEQUENTIAL);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AecProcessor");

#ifdef _WIN32
        int wide_size = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
        std::wstring wide_path(wide_size, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wide_path[0], wide_size);
        Ort::Session session(*env_, wide_path.c_str(), session_options);
#else
        Ort::Session session(*env_, model_path.c_str(), session_options);
#endif
        session_ = std::make_shared<Ort::Session>(std::move(session));

        input_names_  = session_->GetInputNames();
        output_names_ = session_->GetOutputNames();
    }

    ~AecProcessor() {
        if (fft_plan_) { pffft_destroy_setup(fft_plan_); }
        if (fft_in_)   { pffft_aligned_free(fft_in_); }
        if (fft_out_)  { pffft_aligned_free(fft_out_); }
        if (ifft_out_) { pffft_aligned_free(ifft_out_); }
    }

    void process_frame(const float* mic_480, const float* far_480, float* output_480) {
        std::memmove(mic_buffer_.data(), mic_buffer_.data() + AEC_HOP,
                     (AEC_WIN - AEC_HOP) * sizeof(float));
        std::memcpy(mic_buffer_.data() + AEC_WIN - AEC_HOP, mic_480, AEC_HOP * sizeof(float));

        std::memmove(far_buffer_.data(), far_buffer_.data() + AEC_HOP,
                     (AEC_WIN - AEC_HOP) * sizeof(float));
        std::memcpy(far_buffer_.data() + AEC_WIN - AEC_HOP, far_480, AEC_HOP * sizeof(float));

        compute_stft_frame(mic_buffer_.data(), mic_onnx_.data());
        compute_stft_frame(far_buffer_.data(), far_onnx_.data());

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<int64_t> spec_shape = {1, 2, AEC_FREQ};
        std::vector<int64_t> flat2d_1   = {1, 0};
        std::vector<int64_t> prev_shape  = {1, 1, 1, PREV_SIZE};

        std::vector<Ort::Value> inputs;
        inputs.reserve(NUM_INPUTS);

        flat2d_1[1] = AEC_FREQ * 2;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, mic_onnx_.data(), mic_onnx_.size(),
            spec_shape.data(), spec_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, far_onnx_.data(), far_onnx_.size(),
            spec_shape.data(), spec_shape.size()));

        flat2d_1[1] = RES_ENC_CONV_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, res_enc_conv_.data(), res_enc_conv_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = RES_ENC_TFA_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, res_enc_tfa_.data(), res_enc_tfa_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = MIC_ENC_CONV_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, mic_enc_conv_.data(), mic_enc_conv_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = MIC_ENC_TFA_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, mic_enc_tfa_.data(), mic_enc_tfa_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = DEEP_ENC_TFA_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, deep_enc_tfa_.data(), deep_enc_tfa_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = DEC_CONV_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, dec_conv_.data(), dec_conv_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = DEC_TFA_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, dec_tfa_.data(), dec_tfa_.size(),
            flat2d_1.data(), flat2d_1.size()));
        flat2d_1[1] = INTER_SIZE;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, inter_.data(), inter_.size(),
            flat2d_1.data(), flat2d_1.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, res_prev1_.data(), res_prev1_.size(),
            prev_shape.data(), prev_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, res_prev2_.data(), res_prev2_.size(),
            prev_shape.data(), prev_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, mic_prev1_.data(), mic_prev1_.size(),
            prev_shape.data(), prev_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, mic_prev2_.data(), mic_prev2_.size(),
            prev_shape.data(), prev_shape.size()));

        std::vector<const char*> in_names(input_names_.size());
        for (size_t i = 0; i < input_names_.size(); ++i)
            in_names[i] = input_names_[i].c_str();
        std::vector<const char*> out_names(output_names_.size());
        for (size_t i = 0; i < output_names_.size(); ++i)
            out_names[i] = output_names_[i].c_str();

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     in_names.data(), inputs.data(), inputs.size(),
                                     out_names.data(), out_names.size());

        copy_cache_output(outputs[1],  res_enc_conv_,  RES_ENC_CONV_SIZE);
        copy_cache_output(outputs[2],  res_enc_tfa_,   RES_ENC_TFA_SIZE);
        copy_cache_output(outputs[3],  mic_enc_conv_,  MIC_ENC_CONV_SIZE);
        copy_cache_output(outputs[4],  mic_enc_tfa_,   MIC_ENC_TFA_SIZE);
        copy_cache_output(outputs[6],  deep_enc_tfa_,  DEEP_ENC_TFA_SIZE);
        copy_cache_output(outputs[7],  dec_conv_,      DEC_CONV_SIZE);
        copy_cache_output(outputs[8],  dec_tfa_,       DEC_TFA_SIZE);
        copy_cache_output(outputs[9],  inter_,         INTER_SIZE);
        copy_cache_output(outputs[10], res_prev1_,     PREV_SIZE);
        copy_cache_output(outputs[11], res_prev2_,     PREV_SIZE);
        copy_cache_output(outputs[12], mic_prev1_,     PREV_SIZE);
        copy_cache_output(outputs[13], mic_prev2_,     PREV_SIZE);

        float* enhanced_data = outputs[0].GetTensorMutableData<float>();

        std::vector<float> enhanced_interleaved(AEC_FREQ * 2, 0.0f);
        for (int k = 0; k < AEC_FREQ; ++k) {
            enhanced_interleaved[k * 2]     = enhanced_data[k];
            enhanced_interleaved[k * 2 + 1] = enhanced_data[AEC_FREQ + k];
        }

        fft_out_[0] = enhanced_interleaved[0];
        fft_out_[1] = enhanced_interleaved[(AEC_FREQ - 1) * 2];
        for (int k = 1; k < AEC_FREQ - 1; ++k) {
            int pffft_idx = 2 + (k - 1) * 2;
            fft_out_[pffft_idx]     = enhanced_interleaved[k * 2];
            fft_out_[pffft_idx + 1] = enhanced_interleaved[k * 2 + 1];
        }

        pffft_transform_ordered(fft_plan_, fft_out_, ifft_out_, nullptr, PFFFT_BACKWARD);

        float scale = 1.0f / AEC_NFFT;
        for (int i = 0; i < AEC_NFFT; ++i) {
            ifft_out_[i] *= scale * window_[i];
        }

        for (int i = 0; i < AEC_NFFT; ++i) {
            ola_accumulator_[i] += ifft_out_[i];
        }

        for (int i = 0; i < AEC_NFFT; ++i) {
            window_sum_[i] += window_[i] * window_[i];
        }

        for (int i = 0; i < AEC_HOP; ++i) {
            float norm = window_sum_[i];
            output_480[i] = (norm > 1e-6f) ? (ola_accumulator_[i] / norm) : ola_accumulator_[i];
        }

        for (int i = 0; i < AEC_NFFT - AEC_HOP; ++i) {
            ola_accumulator_[i] = ola_accumulator_[i + AEC_HOP];
            window_sum_[i] = window_sum_[i + AEC_HOP];
        }
        for (int i = AEC_NFFT - AEC_HOP; i < AEC_NFFT; ++i) {
            ola_accumulator_[i] = 0.0f;
            window_sum_[i] = 0.0f;
        }
    }

    std::vector<float> process_frame_py(const std::vector<float>& mic_vec, const std::vector<float>& far_vec) {
        std::vector<float> output(AEC_HOP);
        process_frame(mic_vec.data(), far_vec.data(), output.data());
        return output;
    }

    void reset() {
        std::fill(res_enc_conv_.begin(), res_enc_conv_.end(), 0.0f);
        std::fill(res_enc_tfa_.begin(),  res_enc_tfa_.end(),  0.0f);
        std::fill(mic_enc_conv_.begin(), mic_enc_conv_.end(), 0.0f);
        std::fill(mic_enc_tfa_.begin(),  mic_enc_tfa_.end(),  0.0f);
        std::fill(deep_enc_tfa_.begin(), deep_enc_tfa_.end(), 0.0f);
        std::fill(dec_conv_.begin(),     dec_conv_.end(),     0.0f);
        std::fill(dec_tfa_.begin(),      dec_tfa_.end(),      0.0f);
        std::fill(inter_.begin(),        inter_.end(),        0.0f);
        std::fill(res_prev1_.begin(),    res_prev1_.end(),    0.0f);
        std::fill(res_prev2_.begin(),    res_prev2_.end(),    0.0f);
        std::fill(mic_prev1_.begin(),    mic_prev1_.end(),    0.0f);
        std::fill(mic_prev2_.begin(),    mic_prev2_.end(),    0.0f);
        std::fill(mic_buffer_.begin(),   mic_buffer_.end(),   0.0f);
        std::fill(far_buffer_.begin(),   far_buffer_.end(),   0.0f);
        std::fill(ola_accumulator_.begin(), ola_accumulator_.end(), 0.0f);
        std::fill(window_sum_.begin(),   window_sum_.end(),   0.0f);
    }

private:
    static void copy_cache_output(Ort::Value& tensor, std::vector<float>& dst, int expected_size) {
        auto info = tensor.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        if (n == 0) return;
        std::memcpy(dst.data(), tensor.GetTensorMutableData<float>(), expected_size * sizeof(float));
    }

    void compute_stft_frame(const float* input_960, float* onnx_962) {
        for (int i = 0; i < AEC_NFFT; ++i) {
            fft_in_[i] = input_960[i] * window_[i];
        }

        pffft_transform_ordered(fft_plan_, fft_in_, fft_out_, nullptr, PFFFT_FORWARD);

        onnx_962[0] = fft_out_[0];
        onnx_962[AEC_FREQ] = 0.0f;

        for (int k = 1; k < AEC_FREQ - 1; ++k) {
            int pffft_idx = 2 + (k - 1) * 2;
            onnx_962[k]            = fft_out_[pffft_idx];
            onnx_962[AEC_FREQ + k] = fft_out_[pffft_idx + 1];
        }

        onnx_962[AEC_FREQ - 1]            = fft_out_[1];
        onnx_962[AEC_FREQ + AEC_FREQ - 1] = 0.0f;
    }

    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::vector<float> window_;
    float *fft_in_, *fft_out_, *ifft_out_;
    PFFFT_Setup* fft_plan_ = nullptr;

    std::vector<float> mic_buffer_;
    std::vector<float> far_buffer_;

    std::vector<float> ola_accumulator_;
    std::vector<float> window_sum_;

    std::vector<float> mic_onnx_;
    std::vector<float> far_onnx_;

    std::vector<float> res_enc_conv_;
    std::vector<float> res_enc_tfa_;
    std::vector<float> mic_enc_conv_;
    std::vector<float> mic_enc_tfa_;
    std::vector<float> deep_enc_tfa_;
    std::vector<float> dec_conv_;
    std::vector<float> dec_tfa_;
    std::vector<float> inter_;
    std::vector<float> res_prev1_;
    std::vector<float> res_prev2_;
    std::vector<float> mic_prev1_;
    std::vector<float> mic_prev2_;
};

PYBIND11_MODULE(aec_inference, m) {
    #ifdef _WIN32
    HMODULE hModule = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&PyInit_aec_inference, &hModule);
    if (hModule != NULL) {
        char module_path[MAX_PATH];
        GetModuleFileNameA(hModule, module_path, MAX_PATH);
        char* last_slash = strrchr(module_path, '\\');
        if (last_slash != NULL) {
            *last_slash = '\0';

            char onnxruntime_path[MAX_PATH];
            snprintf(onnxruntime_path, MAX_PATH, "%s\\onnxruntime.dll", module_path);

            if (LoadLibraryA(onnxruntime_path) == NULL) {
                snprintf(onnxruntime_path, MAX_PATH, "%s\\cpp\\onnxruntime-win-x64-1.24.4\\lib\\onnxruntime.dll", module_path);
                if (LoadLibraryA(onnxruntime_path) == NULL) {
                    LoadLibraryA("onnxruntime.dll");
                }
            }
        }
    }
    #endif

    py::class_<AecProcessor>(m, "AecProcessor")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("process_frame", &AecProcessor::process_frame_py)
        .def("reset", &AecProcessor::reset);
}
