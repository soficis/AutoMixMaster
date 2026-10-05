#include "ai/OnnxExternalData.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <system_error>

namespace automix::ai {
namespace {

// onnx.proto field numbers this rewrite touches. Everything else is opaque.
constexpr std::uint32_t kModelGraph = 7;          // ModelProto.graph
constexpr std::uint32_t kGraphInitializer = 5;    // GraphProto.initializer
constexpr std::uint32_t kTensorRawData = 9;       // TensorProto.raw_data
constexpr std::uint32_t kTensorExternalData = 13; // TensorProto.external_data
constexpr std::uint32_t kTensorDataLocation = 14; // TensorProto.data_location
constexpr std::uint64_t kDataLocationExternal = 1;
constexpr std::uint32_t kEntryKey = 1;            // StringStringEntryProto.key
constexpr std::uint32_t kEntryValue = 2;          // StringStringEntryProto.value

constexpr std::uint32_t kWireVarint = 0;
constexpr std::uint32_t kWire64 = 1;
constexpr std::uint32_t kWireLength = 2;
constexpr std::uint32_t kWire32 = 5;

// Protobuf refuses messages of 2 GB or more.
constexpr std::uint64_t kMaxMessageBytes = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());

struct Field {
  std::uint32_t number = 0;
  std::uint32_t wire = 0;
  std::size_t begin = 0;         // first byte of the tag
  std::size_t end = 0;           // one past the last byte of the field
  std::size_t payloadBegin = 0;  // length-delimited payload
  std::size_t payloadEnd = 0;
  std::uint64_t varint = 0;
};

bool readVarint(const std::string& bytes, std::size_t& pos, std::size_t end, std::uint64_t& value) {
  value = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (pos >= end) {
      return false;
    }
    const auto byte = static_cast<std::uint8_t>(bytes[pos++]);
    value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      return true;
    }
  }
  return false;
}

void writeVarint(std::string& out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void writeLengthDelimitedHeader(std::string& out, std::uint32_t number, std::uint64_t length) {
  writeVarint(out, (static_cast<std::uint64_t>(number) << 3) | kWireLength);
  writeVarint(out, length);
}

// Reads the field starting at `pos` (which must be < end) and advances past it.
bool nextField(const std::string& bytes, std::size_t& pos, std::size_t end, Field& field, std::string& error) {
  field = Field{};
  field.begin = pos;
  std::uint64_t tag = 0;
  if (!readVarint(bytes, pos, end, tag)) {
    error = "truncated field tag at byte " + std::to_string(field.begin);
    return false;
  }
  field.number = static_cast<std::uint32_t>(tag >> 3);
  field.wire = static_cast<std::uint32_t>(tag & 0x7);
  switch (field.wire) {
    case kWireVarint:
      if (!readVarint(bytes, pos, end, field.varint)) {
        error = "truncated varint at byte " + std::to_string(field.begin);
        return false;
      }
      break;
    case kWire64:
    case kWire32: {
      const std::size_t width = field.wire == kWire64 ? 8 : 4;
      if (end - pos < width) {
        error = "truncated fixed-width field at byte " + std::to_string(field.begin);
        return false;
      }
      pos += width;
      break;
    }
    case kWireLength: {
      std::uint64_t length = 0;
      if (!readVarint(bytes, pos, end, length) || length > end - pos) {
        error = "length-delimited field overruns its message at byte " + std::to_string(field.begin);
        return false;
      }
      field.payloadBegin = pos;
      field.payloadEnd = pos + static_cast<std::size_t>(length);
      pos = field.payloadEnd;
      break;
    }
    default:
      error = "unsupported protobuf wire type " + std::to_string(field.wire) + " at byte " +
              std::to_string(field.begin);
      return false;
  }
  field.end = pos;
  return true;
}

// Refuses absolute locations and any that leave the model's directory.
bool resolveLocation(const std::filesystem::path& modelDirectory,
                     const std::string& location,
                     std::filesystem::path& resolved,
                     std::string& error) {
  const auto relative = std::filesystem::path(location);
  if (location.empty() || relative.is_absolute() || relative.has_root_name() || relative.has_root_directory()) {
    error = "external data location '" + location + "' is not a relative path";
    return false;
  }
  for (const auto& part : relative) {
    if (part == "..") {
      error = "external data location '" + location + "' leaves the model directory";
      return false;
    }
  }
  resolved = modelDirectory / relative;
  return true;
}

class SidecarReader {
 public:
  explicit SidecarReader(std::filesystem::path directory) : directory_(std::move(directory)) {}

  bool read(const std::string& location, std::uint64_t offset, std::uint64_t length, bool lengthGiven,
            std::string& out, std::string& error) {
    auto it = files_.find(location);
    if (it == files_.end()) {
      std::filesystem::path resolved;
      if (!resolveLocation(directory_, location, resolved, error)) {
        return false;
      }
      auto stream = std::make_unique<std::ifstream>(resolved, std::ios::binary);
      if (!stream->is_open()) {
        error = "cannot open external data file '" + resolved.string() + "'";
        return false;
      }
      std::error_code sizeError;
      const auto size = std::filesystem::file_size(resolved, sizeError);
      if (sizeError) {
        error = "cannot size external data file '" + resolved.string() + "'";
        return false;
      }
      it = files_.emplace(location, Open{std::move(stream), size}).first;
    }
    auto& file = it->second;
    if (offset > file.size) {
      error = "external data offset " + std::to_string(offset) + " is past the end of '" + location + "'";
      return false;
    }
    const auto available = file.size - offset;
    const auto count = lengthGiven ? length : available;
    if (count > available) {
      error = "external data for '" + location + "' needs " + std::to_string(count) + " bytes at offset " +
              std::to_string(offset) + " but only " + std::to_string(available) + " remain";
      return false;
    }
    const auto start = out.size();
    out.resize(start + static_cast<std::size_t>(count));
    file.stream->seekg(static_cast<std::streamoff>(offset));
    file.stream->read(out.data() + start, static_cast<std::streamsize>(count));
    if (!*file.stream) {
      error = "short read from external data file '" + location + "'";
      return false;
    }
    return true;
  }

 private:
  struct Open {
    std::unique_ptr<std::ifstream> stream;
    std::uintmax_t size = 0;
  };
  std::filesystem::path directory_;
  std::map<std::string, Open> files_;
};

bool parseUnsigned(const std::string& text, std::uint64_t& value) {
  if (text.empty()) {
    return false;
  }
  value = 0;
  for (const char ch : text) {
    if (ch < '0' || ch > '9') {
      return false;
    }
    const auto digit = static_cast<std::uint64_t>(ch - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      return false;
    }
    value = value * 10 + digit;
  }
  return true;
}

// Appends the rewritten TensorProto in [begin, end) to `out`.
bool rewriteTensor(const std::string& bytes, std::size_t begin, std::size_t end, SidecarReader& sidecars,
                   std::string& out, bool& inlined, std::string& error) {
  std::map<std::string, std::string> external;
  bool isExternal = false;
  std::string kept;
  for (std::size_t pos = begin; pos < end;) {
    Field field;
    if (!nextField(bytes, pos, end, field, error)) {
      return false;
    }
    if (field.number == kTensorDataLocation && field.wire == kWireVarint) {
      isExternal = field.varint == kDataLocationExternal;
      continue;
    }
    if (field.number == kTensorExternalData && field.wire == kWireLength) {
      std::string key;
      std::string value;
      for (std::size_t entry = field.payloadBegin; entry < field.payloadEnd;) {
        Field part;
        if (!nextField(bytes, entry, field.payloadEnd, part, error)) {
          return false;
        }
        if (part.wire != kWireLength) {
          continue;
        }
        auto text = bytes.substr(part.payloadBegin, part.payloadEnd - part.payloadBegin);
        if (part.number == kEntryKey) {
          key = std::move(text);
        } else if (part.number == kEntryValue) {
          value = std::move(text);
        }
      }
      external[key] = value;
      continue;
    }
    kept.append(bytes, field.begin, field.end - field.begin);
  }

  inlined = false;
  if (!isExternal) {
    out.append(bytes, begin, end - begin);  // untouched, byte for byte
    return true;
  }

  const auto location = external.find("location");
  if (location == external.end()) {
    error = "external tensor has no 'location' entry";
    return false;
  }
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
  bool lengthGiven = false;
  if (const auto it = external.find("offset"); it != external.end() && !parseUnsigned(it->second, offset)) {
    error = "external tensor has a malformed offset '" + it->second + "'";
    return false;
  }
  if (const auto it = external.find("length"); it != external.end()) {
    if (!parseUnsigned(it->second, length)) {
      error = "external tensor has a malformed length '" + it->second + "'";
      return false;
    }
    lengthGiven = true;
  }

  std::string raw;
  if (!sidecars.read(location->second, offset, length, lengthGiven, raw, error)) {
    return false;
  }
  out.append(kept);
  writeLengthDelimitedHeader(out, kTensorRawData, raw.size());
  out.append(raw);
  inlined = true;
  return true;
}

bool rewriteGraph(const std::string& bytes, std::size_t begin, std::size_t end, SidecarReader& sidecars,
                  std::string& out, int& inlinedCount, std::string& error) {
  for (std::size_t pos = begin; pos < end;) {
    Field field;
    if (!nextField(bytes, pos, end, field, error)) {
      return false;
    }
    if (field.number != kGraphInitializer || field.wire != kWireLength) {
      out.append(bytes, field.begin, field.end - field.begin);
      continue;
    }
    std::string tensor;
    bool inlined = false;
    if (!rewriteTensor(bytes, field.payloadBegin, field.payloadEnd, sidecars, tensor, inlined, error)) {
      return false;
    }
    inlinedCount += inlined ? 1 : 0;
    writeLengthDelimitedHeader(out, kGraphInitializer, tensor.size());
    out.append(tensor);
    if (out.size() > kMaxMessageBytes) {
      error = "inlined model would exceed protobuf's 2 GB limit; keep its external data";
      return false;
    }
  }
  return true;
}

bool readWholeFile(const std::filesystem::path& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  in.seekg(0, std::ios::end);
  out.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0, std::ios::beg);
  in.read(out.data(), static_cast<std::streamsize>(out.size()));
  return static_cast<bool>(in);
}

} // namespace

ExternalDataInlineResult inlineExternalData(const std::filesystem::path& modelPath,
                                            const std::filesystem::path& outputPath) {
  ExternalDataInlineResult result;
  std::string model;
  if (!readWholeFile(modelPath, model)) {
    result.error = "cannot read model '" + modelPath.string() + "'";
    return result;
  }

  SidecarReader sidecars(modelPath.parent_path());
  std::string rewritten;
  bool sawGraph = false;
  for (std::size_t pos = 0; pos < model.size();) {
    Field field;
    if (!nextField(model, pos, model.size(), field, result.error)) {
      result.error = "'" + modelPath.filename().string() + "' is not a valid ONNX model: " + result.error;
      return result;
    }
    if (field.number != kModelGraph || field.wire != kWireLength) {
      rewritten.append(model, field.begin, field.end - field.begin);
      continue;
    }
    sawGraph = true;
    std::string graph;
    if (!rewriteGraph(model, field.payloadBegin, field.payloadEnd, sidecars, graph, result.tensorsInlined,
                      result.error)) {
      return result;
    }
    writeLengthDelimitedHeader(rewritten, kModelGraph, graph.size());
    rewritten.append(graph);
  }
  if (!sawGraph) {
    result.error = "'" + modelPath.filename().string() + "' has no graph";
    return result;
  }
  if (rewritten.size() > kMaxMessageBytes) {
    result.error = "inlined model would exceed protobuf's 2 GB limit; keep its external data";
    return result;
  }

  auto temporary = outputPath;
  temporary += ".partial";
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out.write(rewritten.data(), static_cast<std::streamsize>(rewritten.size()));
    if (!out) {
      result.error = "cannot write '" + temporary.string() + "'";
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      return result;
    }
  }
  std::error_code renameError;
  std::filesystem::rename(temporary, outputPath, renameError);
  if (renameError) {
    result.error = "cannot move inlined model into place: " + renameError.message();
    std::filesystem::remove(temporary, renameError);
    return result;
  }
  result.success = true;
  return result;
}

} // namespace automix::ai
