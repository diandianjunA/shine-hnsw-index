#include "timing.hh"

#include <ctime>
#include <iomanip>
#include <library/utils.hh>

namespace timing {

Timing::Interval::Interval(str&& descriptor) : descriptor_(std::forward<str>(descriptor)) {
  clock_id_ = CLOCK_MONOTONIC;
  clear();
}

void Timing::Interval::start() {
  lib_assert(clock_gettime(clock_id_, &time_start_) == 0, "calling clock_gettime failed");
}

void Timing::Interval::stop() {
  timespec time_now;  // NOLINT
  lib_assert(clock_gettime(clock_id_, &time_now) == 0, "calling clock_gettime failed");

  time_ = time_now - time_start_ + time_;
}

void Timing::Interval::clear() {
  time_.tv_sec = time_.tv_nsec = 0;
}

void Timing::Interval::add(const IntervalPtr& t2) {
  time_ = time_ + t2->time_;
}

f64 Timing::Interval::get_ms() const {
  return time_.tv_nsec / 1000000.0 + time_.tv_sec * 1000.0;  // NOLINT
}

Timing::IntervalPtr Timing::create_enroll(str&& descriptor) {
  auto interval = std::make_shared<Interval>(std::move(descriptor));
  intervals_.push_back(interval);

  return interval;
}

Timing::json Timing::to_json() const {
  json out;

  for (auto& interval : intervals_) {
    out[interval->descriptor_] = interval->get_ms();
  }

  return out;
}

std::ostream& operator<<(std::ostream& os, const Timing& timing) {
  return os << timing.to_json().dump();
}

timespec operator+(const timespec& ts1, const timespec& ts2) {
  timespec res;  // NOLINT

  if (ts1.tv_sec >= 0 && ts2.tv_sec >= 0) {
    res.tv_sec = ts1.tv_sec + ts2.tv_sec;
    res.tv_nsec = ts1.tv_nsec + ts2.tv_nsec;

    if (res.tv_nsec > 1000000000) {
      res.tv_nsec -= 1000000000;
      res.tv_sec += 1;
    }
  } else {
    std::perror("timing call to operator+ failed");
    std::abort();
  }

  return res;
}

timespec operator-(const timespec& ts1, const timespec& ts2) {
  struct timespec res;  // NOLINT

  if (ts1.tv_sec >= 0 && ts2.tv_sec >= 0) {
    res.tv_sec = ts1.tv_sec - ts2.tv_sec;
    res.tv_nsec = ts1.tv_nsec - ts2.tv_nsec;

    if (res.tv_nsec < 0) {
      res.tv_nsec += 1000000000;
      res.tv_sec -= 1;
    }
  } else {
    std::perror("timing call to operator- failed");
    std::abort();
  }

  return res;
}

std::ostream& operator<<(std::ostream& os, const timespec& ts) {
  os << ts.tv_sec << ".";
  const char prev = os.fill('0');
  os << std::setw(9) << ts.tv_nsec;
  os.fill(prev);

  return os;
}

nlohmann::json get_timestamp() {
  nlohmann::json json_obj;
  const std::time_t now = std::time(nullptr);
  const std::tm* time = std::localtime(&now);

  json_obj["$date"] = (std::stringstream{} << std::put_time(time, "%Y-%m-%dT%H:%M:%SZ")).str();
  return json_obj;
}

}  // namespace timing
