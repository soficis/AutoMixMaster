#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace automix::devtools {

using CommandArgs = std::vector<std::string>;
using CommandFn = std::function<int(const CommandArgs&)>;

class CommandRegistry {
 public:
  void add(const std::string& name, CommandFn fn) {
    commands_[name] = std::move(fn);
  }

  int dispatch(const std::string& name, const CommandArgs& args) const {
    const auto it = commands_.find(name);
    if (it == commands_.end()) {
      return -1;
    }
    return it->second(args);
  }

  std::vector<std::string> commandNames() const {
    std::vector<std::string> names;
    names.reserve(commands_.size());
    for (const auto& [name, _] : commands_) {
      names.push_back(name);
    }
    return names;
  }

 private:
  std::map<std::string, CommandFn> commands_;
};

} // namespace automix::devtools
