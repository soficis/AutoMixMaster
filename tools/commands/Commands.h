#pragma once

namespace automix::devtools {
class CommandRegistry;
}

void registerModelCommands(automix::devtools::CommandRegistry& registry);
void registerSessionCommands(automix::devtools::CommandRegistry& registry);
void registerRenderCommands(automix::devtools::CommandRegistry& registry);
void registerEvalCommands(automix::devtools::CommandRegistry& registry);
void registerBatchCommands(automix::devtools::CommandRegistry& registry);
