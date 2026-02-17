#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <iostream>

#include <juce_core/juce_core.h>

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
  juce::StringArray command;
  command.add(juce::String(python));
  command.add(juce::String(scriptPath->string()));
  juce::StringArray displayCommand = command;

  if (const auto host = argValue(args, "--host"); host.has_value()) {
    command.add("--host");
    command.add(juce::String(*host));
    displayCommand.add("--host");
    displayCommand.add(juce::String(*host));
  }
  if (const auto port = argValue(args, "--port"); port.has_value()) {
    command.add("--port");
    command.add(juce::String(*port));
    displayCommand.add("--port");
    displayCommand.add(juce::String(*port));
  }
  if (const auto bin = argValue(args, "--automix-bin"); bin.has_value()) {
    command.add("--automix-bin");
    command.add(juce::String(*bin));
    displayCommand.add("--automix-bin");
    displayCommand.add(juce::String(*bin));
  }
  if (const auto outputRoot = argValue(args, "--output-root"); outputRoot.has_value()) {
    command.add("--output-root");
    command.add(juce::String(*outputRoot));
    displayCommand.add("--output-root");
    displayCommand.add(juce::String(*outputRoot));
  }
  if (const auto apiKey = argValue(args, "--api-key"); apiKey.has_value()) {
    command.add("--api-key");
    command.add(juce::String(*apiKey));
    displayCommand.add("--api-key");
    displayCommand.add("******");
  }

  std::cout << "Launching Batch Studio API: " << displayCommand.joinIntoString(" ") << "\n";

  juce::ChildProcess process;
  if (!process.start(command)) {
    std::cerr << "Failed to start Batch Studio API process\n";
    return 1;
  }

  const bool finished = process.waitForProcessToFinish(-1);
  if (!finished) {
    process.kill();
    std::cerr << "Batch Studio API process timed out\n";
    return 1;
  }

  const auto output = process.readAllProcessOutput();
  if (output.isNotEmpty()) {
    std::cout << output;
  }

  return process.getExitCode();
}

} // namespace

void registerBatchCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("batch-studio-api", commandBatchStudioApi);
}
