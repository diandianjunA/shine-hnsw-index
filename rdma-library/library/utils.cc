#include "utils.hh"

#include <cmath>
#include <map>

void lib_failure(const str&& message) {
  std::cerr << "[ERROR]: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

std::string get_ip(const str& node_name) {
  std::map<str, str> node_to_ip{
    {"localhost", "127.0.0.1"},
    {"cluster11", "10.60.50.60"},
    {"cluster12", "10.60.50.61"},
    {"cluster13", "10.60.50.62"},
    {"cluster14", "10.60.50.63"},
    {"cluster15", "10.60.50.64"},
    {"cluster16", "10.60.50.65"},
    {"cluster17", "10.60.50.66"},
    {"cluster18", "10.60.50.67"},
    {"cluster19", "10.60.50.68"},
    {"cluster20", "10.60.50.69"},
  };

  lib_assert(node_to_ip.find(node_name) != node_to_ip.end(),
             "Invalid node name: " + node_name);

  return node_to_ip[node_name];
}

f64 compute_throughput(i32 message_size,
                       i32 repeats,
                       Timepoint start,
                       Timepoint end) {
  return message_size / (ToSeconds(end - start).count() / repeats) /
         std::pow(1000, 2);
}

f64 compute_latency(i32 repeats,
                    Timepoint start,
                    Timepoint end,
                    bool is_read_or_atomic) {
  i32 rtt_factor = is_read_or_atomic ? 1 : 2;
  return ToMicroSeconds(end - start).count() / repeats / rtt_factor;
}

void print_status(str&& status) {
  std::cerr << "[STATUS]: " << status << std::endl;
}