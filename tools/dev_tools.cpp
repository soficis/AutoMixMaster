#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "commands/CommandRegistry.h"
#include "commands/Commands.h"

namespace {

void printUsage(const automix::devtools::CommandRegistry& registry) {
  std::cout << "Usage: automix_dev_tools <command> [options]\n\n";
  std::cout << "Available commands:\n";
  auto names = registry.commandNames();
  for (const auto& name : names) {
    std::cout << "  " << name << "\n";
  }
  std::cout << "\nRun a command without arguments for command-specific help.\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    automix::devtools::CommandRegistry registry;
    registerModelCommands(registry);
    registerSessionCommands(registry);
    registerRenderCommands(registry);
    registerEvalCommands(registry);
    registerBatchCommands(registry);

    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }

    if (args.empty()) {
      printUsage(registry);
      return 2;
    }

    const std::string command = args.front();
    const int result = registry.dispatch(command, args);
    if (result == -1) {
      std::cerr << "Unknown command: " << command << "\n\n";
      printUsage(registry);
      return 2;
    }
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Developer tool error: " << error.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Developer tool error: unknown exception\n";
    return 1;
  }
}
