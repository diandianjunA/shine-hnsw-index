#pragma once

#include <library/types.hh>
#include <ostream>

#include "nlohmann/json.hh"

namespace timing {

class Timing {
public:
  struct Interval {
    str descriptor_;

    clockid_t clock_id_;
    timespec time_{};
    timespec time_start_{};

    explicit Interval(str&& descriptor);

    void start();
    void stop();
    void clear();
    void add(const s_ptr<Interval>& t2);

    f64 get_ms() const;
  };

public:
  using IntervalPtr = s_ptr<Interval>;
  using json = nlohmann::json;

  IntervalPtr create_enroll(str&& descriptor);
  static void start(const IntervalPtr& interval) { interval->start(); }
  static void stop(const IntervalPtr& interval) { interval->stop(); }
  static void clear(const IntervalPtr& interval) { interval->clear(); }

  json to_json() const;
  static f64 get_ms(const timespec t) { return t.tv_nsec / 1000000.0 + t.tv_sec * 1000.0; }  // NOLINT
  friend std::ostream& operator<<(std::ostream& os, const Timing& timing);

private:
  vec<IntervalPtr> intervals_;
};

timespec operator+(const timespec& ts1, const timespec& ts2);
timespec operator-(const timespec& ts1, const timespec& ts2);
std::ostream& operator<<(std::ostream& os, const timespec& ts);

nlohmann::json get_timestamp();

}  // namespace timing
