#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <iostream>
#include <sstream>

namespace {

using namespace automix::devtools;

int commandBatchStudioApi(const CommandArgs& args) {
  const auto scriptPath = findRepoPath("tools/batch_studio_api.py");
  if (!scriptPath.has_value()) {
    std::cerr << "batch-studio-api script not found under tools/batch_studio_api.py\n";
    return 1;
  }

  const auto python = argValue(args, "--python")
                          .value_or(readEnvironment("PYTHON").value_or("python3"));
  std::ostringstream command;
  command << python << " \"" << scriptPath->string() << "\"";

  if (const auto host = argValue(args, "--host"); host.has_value()) {
    command << " --host " << *host;
  }
  if (const auto port = argValue(args, "--port"); port.has_value()) {
    command << " --port " << *port;
  }
  if (const auto bin = argValue(args, "--automix-bin"); bin.has_value()) {
    command << " --automix-bin \"" << *bin << "\"";
  }
  if (const auto outputRoot = argValue(args, "--output-root"); outputRoot.has_value()) {
    command << " --output-root \"" << *outputRoot << "\"";
  }
  if (const auto apiKey = argValue(args, "--api-key"); apiKey.has_value()) {
    command << " --api-key \"" << *apiKey << "\"";
  }

  std::cout << "Launching Batch Studio API: " << command.str() << "\n";
  return std::system(command.str().c_str());
}

} // namespace

void registerBatchCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("batch-studio-api", commandBatchStudioApi);
}
