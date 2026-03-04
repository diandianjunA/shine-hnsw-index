#pragma once

#include "common/httplib.h"
#include "common/types.hh"
#include "nlohmann/json.hh"

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace http_server {

using json = nlohmann::json;

struct InsertRequest {
  std::vector<element_t> vector;
  node_t id;
};

struct QueryRequest {
  std::vector<element_t> vector;
  u32 k;
  u32 ef_search;
};

struct RequestTask {
  enum Type { INSERT, QUERY };
  Type type;
  InsertRequest insert_req;
  QueryRequest query_req;
  std::promise<json> promise;
};

class HttpServer {
public:
  HttpServer(const std::string& host, int port);
  ~HttpServer();

  void start();
  void stop();
  bool is_running() const;

  std::optional<RequestTask> get_next_task();
  void submit_response(std::promise<json>& promise, const json& response);

  static constexpr size_t MAX_QUEUE_SIZE = 10000;

private:
  void setup_routes();
  void handle_insert(const httplib::Request& req, httplib::Response& res);
  void handle_query(const httplib::Request& req, httplib::Response& res);
  void handle_health(const httplib::Request& req, httplib::Response& res);
  void handle_info(const httplib::Request& req, httplib::Response& res);

  std::string host_;
  int port_;
  std::unique_ptr<httplib::Server> server_;
  std::thread server_thread_;
  std::atomic<bool> running_{false};

  std::queue<RequestTask> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<size_t> queue_size_{0};
};

}  // namespace http_server
