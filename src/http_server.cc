#include "http_server.hh"

#include <iostream>

namespace http_server {

HttpServer::HttpServer(const std::string& host, int port)
    : host_(host), port_(port), server_(std::make_unique<httplib::Server>()) {
  setup_routes();
}

HttpServer::~HttpServer() {
  stop();
}

void HttpServer::setup_routes() {
  server_->Post("/insert", [this](const httplib::Request& req, httplib::Response& res) {
    handle_insert(req, res);
  });

  server_->Post("/query", [this](const httplib::Request& req, httplib::Response& res) {
    handle_query(req, res);
  });

  server_->Post("/save", [this](const httplib::Request& req, httplib::Response& res) {
    handle_save(req, res);
  });

  server_->Post("/load", [this](const httplib::Request& req, httplib::Response& res) {
    handle_load(req, res);
  });

  server_->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
    handle_health(req, res);
  });

  server_->Get("/info", [this](const httplib::Request& req, httplib::Response& res) {
    handle_info(req, res);
  });

  server_->set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      json error = {{"success", false}, {"error", e.what()}};
      res.set_content(error.dump(), "application/json");
      res.status = 500;
    } catch (...) {
      json error = {{"success", false}, {"error", "Unknown error"}};
      res.set_content(error.dump(), "application/json");
      res.status = 500;
    }
  });
}

void HttpServer::handle_insert(const httplib::Request& req, httplib::Response& res) {
  try {
    if (queue_size_.load() >= MAX_QUEUE_SIZE) {
      json error = {{"success", false}, {"error", "Server busy, please try again later"}};
      res.set_content(error.dump(), "application/json");
      res.status = 503;
      return;
    }

    json body = json::parse(req.body);
    InsertRequest insert_req;
    insert_req.vector = body["vector"].get<std::vector<element_t>>();
    insert_req.id = body.contains("id") ? body["id"].get<node_t>() : static_cast<node_t>(-1);

    RequestTask task;
    task.type = RequestTask::INSERT;
    task.insert_req = insert_req;
    auto future = task.promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      task_queue_.push(std::move(task));
      queue_size_++;
    }
    queue_cv_.notify_one();

    json result = future.get();
    res.set_content(result.dump(), "application/json");
    res.status = result["success"] ? 200 : 400;
  } catch (const std::exception& e) {
    json error = {{"success", false}, {"error", e.what()}};
    res.set_content(error.dump(), "application/json");
    res.status = 400;
  }
}

void HttpServer::handle_query(const httplib::Request& req, httplib::Response& res) {
  try {
    if (queue_size_.load() >= MAX_QUEUE_SIZE) {
      json error = {{"success", false}, {"error", "Server busy, please try again later"}};
      res.set_content(error.dump(), "application/json");
      res.status = 503;
      return;
    }

    json body = json::parse(req.body);
    QueryRequest query_req;
    query_req.vector = body["vector"].get<std::vector<element_t>>();
    query_req.k = body.contains("k") ? body["k"].get<u32>() : 10;
    query_req.ef_search = body.contains("ef_search") ? body["ef_search"].get<u32>() : 100;

    RequestTask task;
    task.type = RequestTask::QUERY;
    task.query_req = query_req;
    auto future = task.promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      task_queue_.push(std::move(task));
      queue_size_++;
    }
    queue_cv_.notify_one();

    json result = future.get();
    res.set_content(result.dump(), "application/json");
    res.status = result["success"] ? 200 : 400;
  } catch (const std::exception& e) {
    json error = {{"success", false}, {"error", e.what()}};
    res.set_content(error.dump(), "application/json");
    res.status = 400;
  }
}

void HttpServer::handle_save(const httplib::Request& req, httplib::Response& res) {
  try {
    json body = json::parse(req.body);
    SaveRequest save_req;
    save_req.path = body.contains("path") ? body["path"].get<std::string>() : "";

    RequestTask task;
    task.type = RequestTask::SAVE;
    task.save_req = save_req;
    auto future = task.promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      task_queue_.push(std::move(task));
      queue_size_++;
    }
    queue_cv_.notify_one();

    json result = future.get();
    res.set_content(result.dump(), "application/json");
    res.status = result["success"] ? 200 : 400;
  } catch (const std::exception& e) {
    json error = {{"success", false}, {"error", e.what()}};
    res.set_content(error.dump(), "application/json");
    res.status = 400;
  }
}

void HttpServer::handle_load(const httplib::Request& req, httplib::Response& res) {
  try {
    json body = json::parse(req.body);
    LoadRequest load_req;
    load_req.path = body.contains("path") ? body["path"].get<std::string>() : "";

    RequestTask task;
    task.type = RequestTask::LOAD;
    task.load_req = load_req;
    auto future = task.promise.get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      task_queue_.push(std::move(task));
      queue_size_++;
    }
    queue_cv_.notify_one();

    json result = future.get();
    res.set_content(result.dump(), "application/json");
    res.status = result["success"] ? 200 : 400;
  } catch (const std::exception& e) {
    json error = {{"success", false}, {"error", e.what()}};
    res.set_content(error.dump(), "application/json");
    res.status = 400;
  }
}

void HttpServer::set_save_callback(SaveCallback cb) {
  save_callback_ = std::move(cb);
}

void HttpServer::set_load_callback(LoadCallback cb) {
  load_callback_ = std::move(cb);
}

void HttpServer::handle_health(const httplib::Request&, httplib::Response& res) {
  json response = {{"status", "ok"}, {"running", running_.load()}, {"success", true}};
  res.set_content(response.dump(), "application/json");
  res.status = 200;
}

void HttpServer::handle_info(const httplib::Request&, httplib::Response& res) {
  json response = {
    {"version", "1.0.0"},
    {"service", "Shine HNSW Vector Store"},
    {"status", "running"}
  };
  res.set_content(response.dump(), "application/json");
  res.status = 200;
}

void HttpServer::start() {
  if (running_.exchange(true)) {
    return;
  }

  server_thread_ = std::thread([this]() {
    std::cerr << "[HTTP Server] Starting on " << host_ << ":" << port_ << std::endl;
    if (!server_->listen(host_.c_str(), port_)) {
      std::cerr << "[HTTP Server] Failed to start server on " << host_ << ":" << port_ << std::endl;
    }
    std::cerr << "[HTTP Server] Listen returned" << std::endl;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void HttpServer::stop() {
  if (!running_.exchange(false)) {
    return;
  }

  server_->stop();
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
  queue_cv_.notify_all();
}

bool HttpServer::is_running() const {
  return running_.load();
}

std::optional<RequestTask> HttpServer::get_next_task() {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  queue_cv_.wait(lock, [this]() {
    return !task_queue_.empty() || !running_;
  });

  if (!running_ && task_queue_.empty()) {
    return std::nullopt;
  }

  RequestTask task = std::move(task_queue_.front());
  task_queue_.pop();
  queue_size_--;
  return task;
}

void HttpServer::submit_response(std::promise<json>& promise, const json& response) {
  promise.set_value(response);
}

}  // namespace http_server
