#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "engine/TransportController.h"

TEST_CASE("TransportController tracks play/seek/advance state", "[transport]") {
  automix::engine::TransportController transport;
  transport.setTimeline(48000, 48000.0);

  REQUIRE(transport.totalSamples() == 48000);
  REQUIRE(transport.progress() == Catch::Approx(0.0));

  transport.play();
  REQUIRE(transport.isPlaying());

  transport.advance(24000);
  REQUIRE(transport.positionSamples() == 24000);
  REQUIRE(transport.progress() == Catch::Approx(0.5).margin(1.0e-6));

  transport.seekToFraction(0.25);
  REQUIRE(transport.positionSamples() == 12000);

  transport.pause();
  REQUIRE_FALSE(transport.isPlaying());

  transport.seekToSample(60000);
  REQUIRE(transport.positionSamples() == 48000);

  transport.stop();
  REQUIRE(transport.positionSamples() == 0);
  REQUIRE(transport.progress() == Catch::Approx(0.0));
}

TEST_CASE("TransportController enforces loop markers during playback", "[transport]") {
  automix::engine::TransportController transport;
  transport.setTimeline(48000, 48000.0);
  transport.setLoopRangeSeconds(0.25, 0.50, true);

  REQUIRE(transport.loopEnabled());
  REQUIRE(transport.loopInSeconds() == Catch::Approx(0.25).margin(1.0e-6));
  REQUIRE(transport.loopOutSeconds() == Catch::Approx(0.50).margin(1.0e-6));

  transport.seekToSample(23500);
  transport.play();
  transport.advance(2000);

  REQUIRE(transport.positionSamples() >= 12000);
  REQUIRE(transport.positionSamples() <= 24000);

  transport.clearLoopRange();
  REQUIRE_FALSE(transport.loopEnabled());
}
