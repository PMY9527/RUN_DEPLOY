// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
// LSTM model has:                                                                                                                                      
//   - Inputs: obs (480), h_in (2×1×256), c_in (2×1×256)                                                                                                                
//   - Outputs: actions (29), h_out (2×1×256), c_out (2×1×256)  
#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

// ONNX Runtime requires exactly one Ort::Env per process
inline Ort::Env& get_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx");
    return env;
}

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        session = std::make_unique<Ort::Session>(get_ort_env(), model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get all output names and shapes
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // First output is actions
        action.resize(output_shapes[0][1]);

        // Initialize hidden states for LSTM (h_in, c_in)
        for (size_t i = 0; i < input_names.size(); ++i) {
            std::string name_str(input_names[i]);
            if (name_str == "h_in" || name_str == "c_in") {
                hidden_states[name_str] = std::vector<float>(input_sizes[i], 0.0f);
                std::cout << "[OrtRunner] Initialized " << name_str << " with size " << input_sizes[i] << std::endl;
            }
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);

            // Use hidden states for h_in/c_in, observations for everything else
            std::vector<float>* input_data_ptr;
            if (hidden_states.find(name_str) != hidden_states.end()) {
                input_data_ptr = &hidden_states[name_str];
            } else {
                if (obs.find(name_str) == obs.end()) {
                    throw std::runtime_error("Input name " + name_str + " not found in observations.");
                }
                input_data_ptr = &obs.at(name_str);
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_ptr->data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        // Copy action output (first output)
        auto action_data = output_tensors[0].GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), action_data, output_shapes[0][1] * sizeof(float));

        // Update hidden states (h_out -> h_in, c_out -> c_in)
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::string output_name_str(output_names[i]);
            if (output_name_str == "h_out") {
                auto h_out_data = output_tensors[i].GetTensorMutableData<float>();
                size_t h_size = 1;
                for (auto dim : output_shapes[i]) h_size *= dim;
                std::memcpy(hidden_states["h_in"].data(), h_out_data, h_size * sizeof(float));
            } else if (output_name_str == "c_out") {
                auto c_out_data = output_tensors[i].GetTensorMutableData<float>();
                size_t c_size = 1;
                for (auto dim : output_shapes[i]) c_size *= dim;
                std::memcpy(hidden_states["c_in"].data(), c_out_data, c_size * sizeof(float));
            }
        }

        return action;
    }

private:
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> input_sizes;

    std::unordered_map<std::string, std::vector<float>> hidden_states;
};

class CMGRunner
{
public:
    CMGRunner(const std::string& model_path, const std::string& data_path)
    {
        
        joints_cmg_to_usd = {0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                             16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28}; // correct
        joints_usd_to_cmg = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11,
                             15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28}; // correct

        // Init ONNX Model
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session = std::make_unique<Ort::Session>(get_ort_env(), model_path.c_str(), session_options);

        // Get input info
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names_owned.push_back(std::move(input_name));
            input_names.push_back(input_names_owned.back().get());

            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
        }

        // Get output info
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names_owned.push_back(std::move(output_name));
            output_names.push_back(output_names_owned.back().get());

            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
        }

        // Load normalization stats
        load_stats(data_path);

        // Pre-allocate buffers
        motion_ref.resize(motion_dim);
        motion_norm.resize(motion_dim);
        cmd_norm.resize(cmd_dim);

        std::cout << "[CMGRunner] Loaded model: " << model_path << std::endl;
        std::cout << "[CMGRunner] motion_dim=" << motion_dim << ", cmd_dim=" << cmd_dim << std::endl;
        std::cout << "[CMGRunner] Input shapes: motion=[" << input_shapes[0][0] << "," << input_shapes[0][1]
                  << "], cmd=[" << input_shapes[1][0] << "," << input_shapes[1][1] << "]" << std::endl;
        std::cout << "[CMGRunner] Output shape: [" << output_shapes[0][0] << "," << output_shapes[0][1] << "]" << std::endl;
    }

    void load_stats(const std::string& data_path)
    {
        YAML::Node stats = YAML::LoadFile(data_path);

        motion_mean = stats["motion_mean"].as<std::vector<float>>();
        motion_std = stats["motion_std"].as<std::vector<float>>();
        cmd_min = stats["command_min"].as<std::vector<float>>();
        cmd_max = stats["command_max"].as<std::vector<float>>();
        cmd_mean = stats["command_mean"].as<std::vector<float>>();
        cmd_std = stats["command_std"].as<std::vector<float>>();

        motion_dim = motion_mean.size();  // 58
        cmd_dim = cmd_min.size();         // 3

        std::cout << "[CMGRunner] Loaded stats from: " << data_path << std::endl;
    }

    // Convert USD order to CMG/SDK order, correct.
    std::vector<float> usd_to_cmg(const std::vector<float>& pos_usd, const std::vector<float>& vel_usd)
    {
        std::vector<float> pos_cmg(29), vel_cmg(29);
        for (size_t i = 0; i < 29; ++i) {
            pos_cmg[i] = pos_usd[joints_usd_to_cmg[i]];
            vel_cmg[i] = vel_usd[joints_usd_to_cmg[i]];
        }
        std::vector<float> motion_cmg(58);
        std::copy(pos_cmg.begin(), pos_cmg.end(), motion_cmg.begin());
        std::copy(vel_cmg.begin(), vel_cmg.end(), motion_cmg.begin() + 29);
        return motion_cmg;
    }

    // Convert CMG/SDK order to USD order, correct.
    std::vector<float> cmg_to_usd(const std::vector<float>& motion_cmg)
    {
        std::vector<float> pos_cmg(motion_cmg.begin(), motion_cmg.begin() + 29);
        std::vector<float> vel_cmg(motion_cmg.begin() + 29, motion_cmg.end());
        std::vector<float> pos_usd(29), vel_usd(29);
        for (size_t i = 0; i < 29; ++i) {
            pos_usd[i] = pos_cmg[joints_cmg_to_usd[i]];
            vel_usd[i] = vel_cmg[joints_cmg_to_usd[i]];
        }
        std::vector<float> motion_usd(58);
        std::copy(pos_usd.begin(), pos_usd.end(), motion_usd.begin());
        std::copy(vel_usd.begin(), vel_usd.end(), motion_usd.begin() + 29);
        return motion_usd;
    }

    // Non-autoregressive forward pass.
    // Uses actual robot state every step.
    std::vector<float> forward(const std::vector<float>& joint_pos_usd,
                               const std::vector<float>& joint_vel_usd,
                               const std::vector<float>& command)
    {
        auto motion_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);

        for (size_t i = 0; i < motion_dim; ++i) {
            float clamped = std::clamp(motion_cmg[i],
                                       motion_mean[i] - 3.0f * motion_std[i],
                                       motion_mean[i] + 3.0f * motion_std[i]);
            motion_norm[i] = (clamped - motion_mean[i]) / motion_std[i];
        }

        for (size_t i = 0; i < cmd_dim; ++i) {
            float range = cmd_max[i] - cmd_min[i];
            if (range > 1e-6f) {
                cmd_norm[i] = (command[i] - cmd_min[i]) / range * 2.0f - 1.0f;
            } else {
                cmd_norm[i] = 0.0f;
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, motion_norm.data(), motion_dim,
            input_shapes[0].data(), input_shapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, cmd_norm.data(), cmd_dim,
            input_shapes[1].data(), input_shapes[1].size()));

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        auto* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> motion_ref_cmg(motion_dim);
        for (size_t i = 0; i < motion_dim; ++i) {
            motion_ref_cmg[i] = output_data[i] * motion_std[i] + motion_mean[i];
        }

        auto motion_ref_usd = cmg_to_usd(motion_ref_cmg);

        std::lock_guard<std::mutex> lock(mtx_);
        motion_ref = motion_ref_usd;
        forward_called = true;
        return motion_ref;
    }

    // AR forward pass.
    // On first call initializes from robot state.
    std::vector<float> forward_ar(const std::vector<float>& joint_pos_usd,
                                   const std::vector<float>& joint_vel_usd,
                                   const std::vector<float>& command)
    {
        if (!ar_initialized) {
            prev_output_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);
            ar_initialized = true;
        }

        for (size_t i = 0; i < motion_dim; ++i) {
            float clamped = std::clamp(prev_output_cmg[i],
                                       motion_mean[i] - 3.0f * motion_std[i],
                                       motion_mean[i] + 3.0f * motion_std[i]);
            motion_norm[i] = (clamped - motion_mean[i]) / motion_std[i];
        }

        // for (size_t i = 0; i < cmd_dim; ++i) {
        //     float range = cmd_max[i] - cmd_min[i];
        //     if (range > 1e-6f) {
        //         cmd_norm[i] = (command[i] - cmd_min[i]) / range * 2.0f - 1.0f;
        //     } else {
        //         cmd_norm[i] = 0.0f;
        //     }
        // } // MIN MAX NORM FOR OLD CMG

        for (size_t i = 0; i < cmd_dim; ++i) {
            cmd_norm[i] = (command[i] - cmd_mean[i]) / cmd_std[i]; // ZSCORE NORM FOR NEW CMG
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, motion_norm.data(), motion_dim,
            input_shapes[0].data(), input_shapes[0].size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, cmd_norm.data(), cmd_dim,
            input_shapes[1].data(), input_shapes[1].size()));

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        auto* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> motion_ref_cmg(motion_dim);
        for (size_t i = 0; i < motion_dim; ++i) {
            motion_ref_cmg[i] = output_data[i] * motion_std[i] + motion_mean[i];
            motion_ref_cmg[i] = std::clamp(motion_ref_cmg[i], -3.14f, 3.14f);
        }

        // Leaky AR: blend CMG output with actual state to prevent
        // the AR chain from getting trapped at clamp fixed points
        // during sharp velocity command transitions.
        constexpr float ar_leak = 0.05f;
        auto actual_cmg = usd_to_cmg(joint_pos_usd, joint_vel_usd);
        for (size_t i = 0; i < motion_dim; ++i) {
            prev_output_cmg[i] = (1.0f - ar_leak) * motion_ref_cmg[i]
                                 + ar_leak * actual_cmg[i];
        }

        auto motion_ref_usd = cmg_to_usd(motion_ref_cmg);

        std::lock_guard<std::mutex> lock(mtx_);
        motion_ref = motion_ref_usd;
        forward_called = true;
        return motion_ref;
    }

    // Get cached motion_ref (thread-safe)
    std::vector<float> get_motion_ref()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return motion_ref;
    }

    std::vector<float> get_qref()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto qref = std::vector<float>(motion_ref.begin(), motion_ref.begin() + motion_dim / 2);
        return qref;
    }

    // Reset AR state (call on episode reset or state transition)
    void reset_ar()
    {
        ar_initialized = false;
        std::cout << "[CMG] Reset" << std::endl;
    }

private:
    // ONNX Runtime
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // Names
    std::vector<Ort::AllocatedStringPtr> input_names_owned;
    std::vector<Ort::AllocatedStringPtr> output_names_owned;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    // Normalization stats
    std::vector<float> motion_mean, motion_std;
    std::vector<float> cmd_min, cmd_max, cmd_mean, cmd_std;
    size_t motion_dim = 58;
    size_t cmd_dim = 3;

    // Buffers
    std::vector<float> motion_ref;  // output [58] in USD order
    std::vector<float> motion_norm; // normalized input in CMG order
    std::vector<float> cmd_norm;    // normalized command

    // Joint order conversion (USD <-> CMG/SDK)
    std::vector<int> joints_usd_to_cmg;
    std::vector<int> joints_cmg_to_usd;

    // Autoregressive state (in CMG order)
    std::vector<float> prev_output_cmg;
    bool ar_initialized = false;

    std::mutex mtx_;
    bool forward_called = false;
};

};