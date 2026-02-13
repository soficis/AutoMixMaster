#include "domain/BatchTypes.h"

namespace automix::domain {

std::string toString(const BatchItemStatus status) {
  switch (status) {
    case BatchItemStatus::Pending:
      return "pending";
    case BatchItemStatus::Analyzing:
      return "analyzing";
    case BatchItemStatus::Rendering:
      return "rendering";
    case BatchItemStatus::Completed:
      return "completed";
    case BatchItemStatus::Failed:
      return "failed";
    case BatchItemStatus::Cancelled:
      return "cancelled";
  }
  return "pending";
}

} // namespace automix::domain
