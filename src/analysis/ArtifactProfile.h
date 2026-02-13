#pragma once

namespace automix::analysis {

struct ArtifactProfile {
  double swirlRisk = 0.0;
  double smearRisk = 0.0;
  double noiseDominance = 0.0;
  double harmonicity = 0.0;
  double phaseInstability = 0.0;
};

} // namespace automix::analysis
