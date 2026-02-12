#pragma once

#include <nlohmann/json.hpp>

#include "domain/Session.h"

namespace automix::domain {

using Json = nlohmann::json;

void to_json(Json& j, const Stem& value);
void from_json(const Json& j, Stem& value);

void to_json(Json& j, const Bus& value);
void from_json(const Json& j, Bus& value);

void to_json(Json& j, const RenderSettings& value);
void from_json(const Json& j, RenderSettings& value);

void to_json(Json& j, const StemMixDecision& value);
void from_json(const Json& j, StemMixDecision& value);

void to_json(Json& j, const MixPlan& value);
void from_json(const Json& j, MixPlan& value);

void to_json(Json& j, const MasterPlan& value);
void from_json(const Json& j, MasterPlan& value);

void to_json(Json& j, const Session& value);
void from_json(const Json& j, Session& value);

} // namespace automix::domain
