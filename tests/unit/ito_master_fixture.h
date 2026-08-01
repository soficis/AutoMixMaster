#pragma once

// Shared ITO-Master test fixture: the real config.json contract consumed by the
// mastering-assistant route (fxencoder.onnx + mastering_tcn.onnx + config.json
// pack artifact). Kept in one place so the adapter and strategy tests assert
// the same static 46-param schema.

#include <filesystem>
#include <fstream>
#include <string>

namespace ito_test {

inline constexpr const char* kConfigFixture = R"json({
  "model": "ITO-Master white-box (params-only ONNX export)",
  "sample_rate": 44100,
  "fx_order": ["eq", "distortion", "multiband_comp", "gain", "imager", "limiter"],
  "num_params": 46,
  "normalized": true,
  "denormalize": "value = norm * (max - min) + min",
  "params": [
    {"fx": "eq", "name": "low_shelf_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "low_shelf_cutoff_freq", "min": 20.0, "max": 2000.0},
    {"fx": "eq", "name": "low_shelf_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "eq", "name": "band0_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "band0_cutoff_freq", "min": 80.0, "max": 2000.0},
    {"fx": "eq", "name": "band0_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "eq", "name": "band1_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "band1_cutoff_freq", "min": 2000.0, "max": 8000.0},
    {"fx": "eq", "name": "band1_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "eq", "name": "band2_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "band2_cutoff_freq", "min": 8000.0, "max": 12000.0},
    {"fx": "eq", "name": "band2_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "eq", "name": "band3_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "band3_cutoff_freq", "min": 12000.0, "max": 21050.0},
    {"fx": "eq", "name": "band3_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "eq", "name": "high_shelf_gain_db", "min": -20.0, "max": 20.0},
    {"fx": "eq", "name": "high_shelf_cutoff_freq", "min": 4000.0, "max": 21050.0},
    {"fx": "eq", "name": "high_shelf_q_factor", "min": 0.1, "max": 5.0},
    {"fx": "distortion", "name": "drive_db", "min": 0.0, "max": 8.0},
    {"fx": "distortion", "name": "parallel_weight_factor", "min": 0.2, "max": 0.7},
    {"fx": "multiband_comp", "name": "low_cutoff", "min": 20.0, "max": 300.0},
    {"fx": "multiband_comp", "name": "high_cutoff", "min": 2000.0, "max": 12000.0},
    {"fx": "multiband_comp", "name": "parallel_weight_factor", "min": 0.2, "max": 0.7},
    {"fx": "multiband_comp", "name": "low_shelf_comp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "low_shelf_comp_ratio", "min": 1.000001, "max": 20.0},
    {"fx": "multiband_comp", "name": "low_shelf_exp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "low_shelf_exp_ratio", "min": 1e-06, "max": 0.999999},
    {"fx": "multiband_comp", "name": "low_shelf_at", "min": 5.0, "max": 100.0},
    {"fx": "multiband_comp", "name": "low_shelf_rt", "min": 5.0, "max": 100.0},
    {"fx": "multiband_comp", "name": "mid_band_comp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "mid_band_comp_ratio", "min": 1.000001, "max": 20.0},
    {"fx": "multiband_comp", "name": "mid_band_exp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "mid_band_exp_ratio", "min": 1e-06, "max": 0.999999},
    {"fx": "multiband_comp", "name": "mid_band_at", "min": 5.0, "max": 100.0},
    {"fx": "multiband_comp", "name": "mid_band_rt", "min": 5.0, "max": 100.0},
    {"fx": "multiband_comp", "name": "high_shelf_comp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "high_shelf_comp_ratio", "min": 1.000001, "max": 20.0},
    {"fx": "multiband_comp", "name": "high_shelf_exp_thresh", "min": -60.0, "max": -1e-06},
    {"fx": "multiband_comp", "name": "high_shelf_exp_ratio", "min": 1e-06, "max": 0.999999},
    {"fx": "multiband_comp", "name": "high_shelf_at", "min": 5.0, "max": 100.0},
    {"fx": "multiband_comp", "name": "high_shelf_rt", "min": 5.0, "max": 100.0},
    {"fx": "gain", "name": "gain_db", "min": -24.0, "max": 24.0},
    {"fx": "imager", "name": "width", "min": 0.0, "max": 1.0},
    {"fx": "limiter", "name": "threshold", "min": -60.0, "max": -1e-06},
    {"fx": "limiter", "name": "at", "min": 5.0, "max": 100.0},
    {"fx": "limiter", "name": "rt", "min": 5.0, "max": 100.0}
  ],
  "graphs": {
    "fxencoder": {"file": "fxencoder.onnx", "in": "ref_audio[1,2,N]", "out": "embedding[1,2048]"},
    "predictor": {"file": "mastering_tcn.onnx", "in": "in_audio[1,2,N]+embedding[1,2048]", "out": "params[1,46]"}
  }
})json";

inline std::filesystem::path writeConfigFixture(const std::string& dirName) {
  const auto dir = std::filesystem::temp_directory_path() / dirName;
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto path = dir / "config.json";
  {
    std::ofstream out(path);
    out << kConfigFixture;
  }
  return path;
}

} // namespace ito_test
