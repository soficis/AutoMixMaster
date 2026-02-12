#pragma once

#include "automix/IAutoMixStrategy.h"

namespace automix::automix {

class HeuristicAutoMixStrategy final : public IAutoMixStrategy {
 public:
  domain::MixPlan buildPlan(const domain::Session& session,
                            const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                            double dryWet) const override;
};

} // namespace automix::automix
