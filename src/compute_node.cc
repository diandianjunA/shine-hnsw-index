#include "compute_node.hh"

#include "cache/placement.hh"
#include "hnsw/distance.hh"
#include "io/read_data.hh"

using json = nlohmann::json;

template <class Distance>
ComputeNode<Distance>::ComputeNode(Configuration& config)
    : context_(config), cm_(context_, config), num_servers_(config.num_server_nodes()) {
  std::cerr << "[DEBUG] ComputeNode constructor started, enable_http=" << config.enable_http << std::endl;
  init_remote_tokens();
  cm_.connect();

  if (!config.disable_thread_pinning) {
    const u32 core = core_assignment_.get_available_core();
    pin_main_thread(core);
    print_status("pinned main thread to core " + std::to_string(core));
  }

  if (cm_.is_initiator) {  // communicate parameters to the memory nodes
    configuration::Parameters p{config.num_threads, config.use_cache, config.routing};

    for (const QP& qp : cm_.server_qps) {
      qp->post_send_inlined(&p, sizeof(configuration::Parameters), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  }

  // enroll timings
  t_build_ = timing_.create_enroll("build_c0");
  t_query_ = timing_.create_enroll("query_c0");

  receive_remote_access_tokens();

  const bool compute_recall = not config.no_recall;
  
  // Read dataset only if not in HTTP mode or loading index
  if (!config.enable_http || config.load_index) {
    read_dataset(config.data_path, config.query_suffix, config.load_index, compute_recall, config.use_cache);
  }

  const u32 seed = config.seed == -1 ? std::random_device{}() : config.seed + cm_.client_id;
  
  // Determine dimension - if in HTTP mode and no dataset, use default 128
  u32 dim = config.enable_http && database_.dim == 0 ? 128 : database_.dim;
  
  hnsw::HNSW<Distance> hnsw{
    config.m, config.ef_construction, config.k, config.ef_search, seed, dim, config.use_cache};

  const size_t num_vectors_estimate = config.enable_http ? 1000000 : database_.num_vectors_total;
  const size_t estimated_index_size = hnsw.estimate_index_size(num_vectors_estimate);
  statistics_.add_static_stat("estimated_total_index_size", estimated_index_size);

  const size_t cache_size = static_cast<f32>(estimated_index_size) / 100. * config.cache_size_ratio;

  if (config.use_cache) {
    print_status("max cache size: " + std::to_string(cache_size));
    statistics_.add_nested_static_stat("cache", "cache_size_ratio", config.cache_size_ratio);
  }

  // ------------------------------------
  // - initialize cache and worker pool -
  // ------------------------------------

  const size_t effective_cache_size = config.use_cache ? cache_size : 0;
  const size_t num_cache_buckets = effective_cache_size > 0 ? effective_cache_size / Node::size_until_components() : 1;
  const size_t num_cooling_table_buckets = effective_cache_size > 0 
      ? std::ceil(effective_cache_size / Node::size_until_components() / cache::COOLING_TABLE_BUCKET_ENTRIES * cache::COOLING_TABLE_RATIO)
      : 1;

  print_status("allocate worker threads and read buffers");
  WorkerPool worker_pool{config.num_threads,
                         config.max_send_queue_wr,
                         effective_cache_size,
                         num_cache_buckets,
                         num_cooling_table_buckets,
                         config.use_cache};

  worker_pool.allocate_worker_threads(context_, cm_, remote_access_tokens_, config.num_coroutines);

  statistics_.add_static_stat("allocated_local_buffer_size", COMPUTE_NODE_MAX_MEMORY);
  statistics_.add_static_stat("distance", config.ip_distance ? "inner_product" : "squared_l2");
  statistics_.add_static_stat("node_size", Node::size_until_components());
  statistics_.add_static_stat("neighborlist_size", Node::NEIGHBORLIST_SIZE);
  statistics_.add_static_stat("neighborlist_size_l0", Node::NEIGHBORLIST_SIZE_ZERO);
  statistics_.add_nested_static_stat("cache", "num_cache_buckets", num_cache_buckets);
  statistics_.add_nested_static_stat("cache", "num_cooling_table_buckets", num_cooling_table_buckets);

  cm_.synchronize();  // notify memory nodes that this compute node is ready

  std::cerr << "[DEBUG] Before enable_http check, enable_http=" << config.enable_http << std::endl;
  if (config.enable_http) {
    // HTTP service mode
    run_http_service_mode(config, hnsw, worker_pool);
  } else {
    // Original benchmark mode
    Placement<Distance> placement{cm_.num_total_clients, worker_pool.get_compute_threads().front(), timing_};
    run_benchmark_mode(config, hnsw, worker_pool, compute_recall, placement);
  }
}

template <class Distance>
void ComputeNode<Distance>::run_benchmark_mode(Configuration& config,
                                                hnsw::HNSW<Distance>& hnsw,
                                                WorkerPool& worker_pool,
                                                const bool compute_recall,
                                                const Placement<Distance>& placement) {
  // construct the index
  if (!config.load_index) {
    run_inserts(hnsw, worker_pool, config.num_coroutines, !config.disable_thread_pinning);
    join_threads(worker_pool.get_compute_threads());
    core_assignment_.get_available_core();  // join_t resets the assignments (but main thread's assignment remains)
    worker_pool.reset_barriers();

    // accumulate build statistics
    for (const auto& thread : worker_pool.get_compute_threads()) {
      cn_statistics_.build_distcomps += thread->stats.distcomps;
      cn_statistics_.build_rdma_reads += thread->stats.rdma_reads_in_bytes;
      cn_statistics_.build_rdma_writes += thread->stats.rdma_writes_in_bytes;
      cn_statistics_.remote_allocations += thread->stats.remote_allocations;
      cn_statistics_.total_allocation_size += thread->stats.allocation_size;
      cn_statistics_.max_level = std::max(cn_statistics_.max_level, thread->stats.max_level);
      thread->reset();
    }

    print_status("deallocate input vectors");
    database_.deallocate();
    sync_compute_nodes();  // do not send store message to memory nodes before all CNs are done constructing the index
  }

  wait_for_load_or_store(config);  // communicate to memory nodes whether to load or store the index (or do nothing)
  sync_compute_nodes();  // start with queries only if all compute nodes are done with constructing/loading the index

  // reset thread statistics and coroutines
  for (const auto& thread : worker_pool.get_compute_threads()) {
    thread->reset();
  }

  // ----------------
  // - cache warmup -
  // ----------------

  if (config.use_cache && cache::CACHE_WARMUP) {
    print_status("cache warmup");
    run_query_router_and_queries(
      warmup_queries_, worker_pool, hnsw, timing_.create_enroll("warmup_routing"), placement, config);

    // reset threads and timer
    t_query_->clear();  // !!!
    warmup_queries_.deallocate();

    for (const auto& thread : worker_pool.get_compute_threads()) {
      thread->reset();
    }

    print_status("cache warmup done");
    std::cerr << "cache full?: " << worker_pool.get_compute_threads().front()->cache.is_full() << std::endl;
  }

  sync_compute_nodes();  // start with queries only if all compute nodes are done with the warmup

  // -------------------
  // - process queries -
  // -------------------

  print_status("run queries");
  run_query_router_and_queries(queries_, worker_pool, hnsw, timing_.create_enroll("routing"), placement, config);

  size_t processed_queries = 0;
  for (const auto& t : worker_pool.get_compute_threads()) {
    processed_queries += t->stats.processed;
  }
  print_status("processed queries: " + std::to_string(processed_queries));

  if (compute_recall) {
    const f64 local_recall = compute_local_recall(worker_pool.get_compute_threads(), config.k, processed_queries);
    print_status("local recall: " + std::to_string(local_recall));

    // distribute recalls to initiator and compute rolling recall:
    // sum_i^n recall_i * (num_queries_processed_i / num_queries_total)
    cn_statistics_.rolling_recall = local_recall * (static_cast<f64>(processed_queries) / queries_.num_vectors_total);
  }

  // accumulate query statistics
  for (const auto& thread : worker_pool.get_compute_threads()) {
    cn_statistics_.query_distcomps += thread->stats.distcomps;
    cn_statistics_.query_rdma_reads += thread->stats.rdma_reads_in_bytes;
    cn_statistics_.query_rdma_writes += thread->stats.rdma_writes_in_bytes;
    cn_statistics_.query_cache_hits += thread->stats.cache_hits;
    cn_statistics_.query_cache_misses += thread->stats.cache_misses;
    cn_statistics_.query_visited_nodes += thread->stats.visited_nodes;
    cn_statistics_.query_visited_nodes_l0 += thread->stats.visited_nodes_l0;
    cn_statistics_.query_visited_neighborlists += thread->stats.visited_neighborlists;
    thread->reset();
  }

  cn_statistics_.processed_queries = processed_queries;
  cn_statistics_.local_allocation_size += worker_pool.get_buffer_allocator().allocated_memory();
  cn_statistics_.build_time = t_build_->time_;
  cn_statistics_.query_time = t_query_->time_;
  collect_statistics_and_timings();

  worker_pool.track_local_cache_statistics(statistics_);

  if (cm_.is_initiator) {
    add_meta_statistics(config);
    statistics_.add_static_stat("num_vectors", database_.num_vectors_total);
    statistics_.add_static_stat("num_queries", queries_.num_vectors_total);
    statistics_.add_nested_static_stat("queries", "compute_recall", compute_recall ? "true" : "false");
    statistics_.output_all(timing_.to_json());

  } else {
    std::cerr << timing_ << std::endl;
  }
}

template <class Distance>
void ComputeNode<Distance>::run_http_service_mode(Configuration& config,
                                                   hnsw::HNSW<Distance>& hnsw,
                                                   WorkerPool& worker_pool) {
  std::cerr << "[DEBUG] run_http_service_mode started" << std::endl;
  print_status("Running in HTTP service mode");
  
  // Initialize HTTP server if we're the initiator
  if (cm_.is_initiator) {
    std::cerr << "[DEBUG] Creating HTTP server" << std::endl;
    http_server_ = std::make_unique<http_server::HttpServer>(config.http_host, config.http_port);
    std::cerr << "[DEBUG] Starting HTTP server" << std::endl;
    http_server_->start();
    print_status("HTTP server started on " + config.http_host + ":" + std::to_string(config.http_port));
    
    // Load index if specified
    if (config.load_index) {
      print_status("Loading index from file");
      wait_for_load_or_store(config);
      sync_compute_nodes();
    }
    
    // Run the HTTP service loop
    run_http_service(hnsw, worker_pool, config.num_coroutines, !config.disable_thread_pinning, config);
    
    // Cleanup
    http_server_->stop();
    terminate();
  } else {
    // Non-initiator compute nodes just wait
    print_status("Non-initiator compute node waiting in HTTP mode");
    if (config.load_index) {
      wait_for_load_or_store(config);
      sync_compute_nodes();
    }
    
    // Wait for termination signal
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      // In a real implementation, we'd have a way to signal termination
      // For now, we just keep this node alive
    }
  }
}

template <class Distance>
void ComputeNode<Distance>::run_query_router_and_queries(io::Database<element_t>& queries,
                                                         WorkerPool& worker_pool,
                                                         hnsw::HNSW<Distance>& hnsw,
                                                         timing::Timing::IntervalPtr&& routing_timer,
                                                         const Placement<Distance>& placement,
                                                         const Configuration& config) {
  // -----------------------------
  // - initialize and run router -
  // -----------------------------

  query_router::QueryRouter query_router{placement,
                                         queries,
                                         context_,
                                         cm_.server_qps,
                                         cm_.num_total_clients,
                                         config.max_send_queue_wr,
                                         cm_.client_id,
                                         std::move(routing_timer)};

  const auto routing_thread = std::make_unique<Thread>(-1);
  if (cm_.num_total_clients > 1 && config.routing) {
    if (cm_.is_initiator) {
      // pass query routing message size to memory nodes
      for (u32 memory_node = 0; memory_node < num_servers_; ++memory_node) {
        const QP& qp = cm_.server_qps[memory_node];

        qp->post_send_inlined(&query_router.message_size, sizeof(size_t), IBV_WR_SEND);
        context_.poll_send_cq_until_completion();
      }
    }

    cm_.synchronize();  // wait until all MNs are ready
    routing_thread->start(&query_router::QueryRouter<Distance>::run_routing, &query_router);

  } else {  // no query routing with just one compute node
    query_router.done = true;
    query_router.queue_size += static_cast<i32>(queries.max_slot);

    for (idx_t slot = 0; slot < queries.max_slot; ++slot) {
      query_router.query_queue.enqueue(slot);
    }
  }

  // -------------------
  // - process queries -
  // -------------------

  run_queries(hnsw, worker_pool, queries, query_router, config.num_coroutines, !config.disable_thread_pinning);
  join_threads(worker_pool.get_compute_threads());
  routing_thread->join();

  core_assignment_.get_available_core();  // join_thread resets the assignments (but main thread's assignment remains)
  worker_pool.reset_barriers();

  terminate();  // notify memory nodes that we are done
}

template <class Distance>
void ComputeNode<Distance>::init_remote_tokens() {
  remote_access_tokens_.resize(num_servers_);

  for (auto& mrt : remote_access_tokens_) {
    mrt = std::make_unique<MemoryRegionToken>();  // ownership has the vector
  }
}

template <class Distance>
void ComputeNode<Distance>::receive_remote_access_tokens() {
  print_status("receive access tokens of remote memory regions");
  for (u32 memory_node = 0; memory_node < num_servers_; ++memory_node) {
    const QP& qp = cm_.server_qps[memory_node];
    MRT& mrt = remote_access_tokens_[memory_node];

    LocalMemoryRegion token_region{context_, mrt.get(), sizeof(MemoryRegionToken)};
    qp->post_receive(token_region);
    context_.receive();
  }
}

/**
 * @brief Reads base vectors (partially), query vectors (partially), and ground truth into Database objects.
 * @param data_path Filepath to the location of `base.fvecs` and the `query` directory.
 * @param query_suffix Filename suffix of the query file.
 * @param load_index If true, index will be read from the memory servers, no need to read it from file (only meta data).
 * @param include_groundtruth If true, also the ground truth will be loaded.
 */
template <class Distance>
void ComputeNode<Distance>::read_dataset(const filepath_t& data_path,
                                         const str& query_suffix,
                                         bool load_index,
                                         bool include_groundtruth,
                                         bool use_cache) {
  filepath_t data_file, query_file, ground_truth_file, warmup_file;

  for (const auto& file : std::filesystem::directory_iterator(data_path)) {
    if (file.path().stem() == "base") {
      data_file = file.path();
      break;
    }
  }

  auto query_path = data_path;
  query_path /= "queries/";

  for (const auto& file : std::filesystem::directory_iterator(query_path)) {
    if (file.path().stem() == str{"query-"} + query_suffix) {
      query_file = file.path();
    } else if (file.path().stem() == str{"groundtruth-"} + query_suffix) {
      ground_truth_file = file.path();
    } else if (file.path().stem() == str{"warmup-"} + query_suffix) {
      warmup_file = file.path();
    }
  }

  lib_assert(not data_file.empty() && not query_file.empty(), "base or query file missing");

  io::read_data_partially<element_t>(database_, data_file, cm_.client_id, cm_.num_total_clients, load_index);
  io::read_data_partially<element_t>(queries_, query_file, cm_.client_id, cm_.num_total_clients);

  if (use_cache && cache::CACHE_WARMUP) {
    lib_assert(not warmup_file.empty(), "warmup file missing");
    io::read_data_partially<element_t>(warmup_queries_, warmup_file, cm_.client_id, cm_.num_total_clients);
  }

  if (include_groundtruth) {
    lib_assert(!ground_truth_file.empty(), "ground truth file missing");
    io::read_data<node_t>(ground_truth_, ground_truth_file);  // read entirely
  }
}

template <class Distance>
void ComputeNode<Distance>::run_inserts(hnsw::HNSW<Distance>& hnsw,
                                        WorkerPool& worker_pool,
                                        u32 num_coroutines,
                                        bool pin_threads) {
  print_status("**INSERT**: running worker threads...");

  for (const auto& t : worker_pool.get_compute_threads()) {
    const u32 thread_id = t->get_id();

    if (thread_id != 0) {
      t->start(&WorkerPool::process_inserts<Distance>,
               &worker_pool,
               std::ref(hnsw),
               std::ref(next_insert_idx_),
               std::ref(database_),
               num_coroutines);

      if (pin_threads) {
        const u32 core = core_assignment_.get_available_core();
        t->set_affinity(core);
        print_status("pinned thread " + std::to_string(thread_id) + " to core " + std::to_string(core));
      }
    }
  }

  // main thread now is also a worker thread and will release the barrier
  t_build_->start();
  worker_pool.process_inserts<Distance>(hnsw, next_insert_idx_, database_, num_coroutines, 0);
  t_build_->stop();
}

template <class Distance>
void ComputeNode<Distance>::run_queries(hnsw::HNSW<Distance>& hnsw,
                                        WorkerPool& worker_pool,
                                        io::Database<element_t>& queries,
                                        query_router::QueryRouter<Distance>& query_router,
                                        u32 num_coroutines,
                                        bool pin_threads) {
  print_status("**QUERY**: running worker threads...");

  for (const auto& t : worker_pool.get_compute_threads()) {
    const u32 thread_id = t->get_id();

    if (thread_id != 0) {
      t->start(&WorkerPool::process_queries<Distance>,
               &worker_pool,
               std::ref(hnsw),
               std::ref(next_query_idx_),
               std::ref(queries),
               std::ref(query_router),
               num_coroutines);

      if (pin_threads) {
        const u32 core = core_assignment_.get_available_core();
        t->set_affinity(core);
        print_status("pinned thread " + std::to_string(thread_id) + " to core " + std::to_string(core));
      }
    }
  }

  // main thread now is also a worker thread and will release the barrier
  t_query_->start();
  worker_pool.process_queries<Distance>(hnsw, next_query_idx_, queries, query_router, num_coroutines, 0);
  t_query_->stop();
}

template <class Distance>
void ComputeNode<Distance>::join_threads(const ComputeThreads& compute_threads) {
  print_status("join compute threads");

  for (const auto& t : compute_threads) {
    // no need for joining the main thread
    if (t->get_id() != 0) {
      t->join();
    }

    // check post balances of coroutines
    for (auto& post_balance : t->post_balances) {
      lib_assert(post_balance == 0, "incomplete READs");
    }

    std::cerr << "t" << t->get_id() << " processed: " << t->stats.processed << ", CHR: " << t->stats.cache_hit_rate()
              << std::endl;
  }

  core_assignment_.reset();
}

/**
 * @brief Send instructions to all memory servers whether to load or store the index (or do nothing).
 *        If load or store: send an individual filepath to each memory server.
 *        Finally, wait until the memory servers are done and send their responses.
 */
template <class Distance>
void ComputeNode<Distance>::wait_for_load_or_store(const Configuration& config) {
  struct Message {
    bool load;
    size_t path_length;
  };

  if (cm_.is_initiator) {
    const size_t num_memory_servers = cm_.server_qps.size();

    // send instructions to all memory servers
    if (config.store_index || config.load_index) {  // both cannot be true
      for (idx_t i = 0; i < num_memory_servers; ++i) {
        filepath_t path = config.data_path;
        path /= "dump/index_m" + std::to_string(config.m) + "_efc" + std::to_string(config.ef_construction) + "_node" +
                std::to_string(i + 1) + "_of" + std::to_string(num_memory_servers) + ".dat";

        const Message msg{config.load_index, path.string().size()};
        const QP& qp = cm_.server_qps[i];

        qp->post_send_inlined(&msg, sizeof(Message), IBV_WR_SEND);
        qp->post_send_inlined(path.string().data(), path.string().size(), IBV_WR_SEND);
      }

      context_.poll_send_cq_until_completion(static_cast<i32>(2 * num_memory_servers));

    } else {  // send null message to indicate that nothing must be done
      for (const auto& qp : cm_.server_qps) {
        constexpr Message msg{false, 0};
        qp->post_send_inlined(&msg, sizeof(Message), IBV_WR_SEND);
      }

      context_.poll_send_cq_until_completion(static_cast<i32>(num_memory_servers));
    }

    const bool success = cm_.synchronize();  // wait until memory nodes respond
    lib_assert(success, "index loading failed");
  }
}

template <class Distance>
void ComputeNode<Distance>::sync_compute_nodes() {
  print_status("synchronize compute nodes");

  if (cm_.is_initiator) {
    bool sync;  // dummy region
    LocalMemoryRegion region{context_, &sync, sizeof(bool)};

    for (const QP& qp : cm_.client_qps) {
      qp->post_receive(region);
    }

    context_.receive(cm_.num_total_clients - 1);

  } else {
    constexpr bool sync = true;  // dummy value

    cm_.initiator_qp->post_send_inlined(&sync, sizeof(bool), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
}

template <class Distance>
void ComputeNode<Distance>::add_meta_statistics(const Configuration& config) {
  const auto path_name = config.data_path.has_stem() ? config.data_path.stem() : config.data_path.parent_path().stem();
  auto zipf_parameter = config.query_suffix.substr(1, config.query_suffix.find_first_of('-') - 1);

  statistics_.add_meta_stats(
    std::make_pair("compute_nodes", cm_.num_total_clients),
    std::make_pair("memory_nodes", num_servers_),
    std::make_pair("compute_threads", cm_.num_total_clients * config.num_threads),
    std::make_pair("coroutines_per_thread", config.num_coroutines),
    std::make_pair("threads_pinned", config.disable_thread_pinning ? "false" : "true"),
    std::make_pair("hyperthreading", core_assignment_.hyperthreading_enabled() ? "true" : "false"),
    std::make_pair("dataset", path_name),
    std::make_pair("query_suffix", config.query_suffix),
    std::make_pair("zipf_parameter", zipf_parameter),
    std::make_pair("timestamp", timing::get_timestamp()),
    std::make_pair("label", config.label));

  statistics_.add_nested_static_stat("hnsw_parameters", "k", config.k);
  statistics_.add_nested_static_stat("hnsw_parameters", "m", config.m);
  statistics_.add_nested_static_stat("hnsw_parameters", "ef_search", config.ef_search);
  statistics_.add_nested_static_stat("hnsw_parameters", "ef_construction", config.ef_construction);
}

template <class Distance>
void ComputeNode<Distance>::collect_statistics_and_timings() {
  print_status("collect statistics and timings from compute nodes");
  const auto chr = [](size_t hits, size_t misses) { return static_cast<f64>(hits) / static_cast<f64>(hits + misses); };

  if (cm_.is_initiator) {
    timespec max_build_time = t_build_->time_;
    timespec max_query_time = t_query_->time_;

    // add local statistics
    statistics_.add_nested_static_stat(
      "cache", "local_hit_rates", "c0", chr(cn_statistics_.query_cache_hits, cn_statistics_.query_cache_misses));
    statistics_.add_nested_static_stat("queries", "processed_local", "c0", cn_statistics_.processed_queries);

    for (u32 client_id = 1; client_id < cm_.num_total_clients; ++client_id) {
      CNStatistics received_stats{};
      LocalMemoryRegion region(context_, &received_stats, sizeof(CNStatistics));

      const QP& qp = cm_.client_qps[client_id - 1];
      qp->post_receive(region);
      context_.receive();

      cn_statistics_.combine(received_stats);

      // add build and query timings of other compute nodes
      const auto build_interval = timing_.create_enroll("build_c" + std::to_string(client_id));
      const auto query_interval = timing_.create_enroll("query_c" + std::to_string(client_id));

      build_interval->time_ = received_stats.build_time;
      query_interval->time_ = received_stats.query_time;

      if (build_interval->get_ms() > timing::Timing::get_ms(max_build_time)) {
        max_build_time = build_interval->time_;
      }
      if (query_interval->get_ms() > timing::Timing::get_ms(max_query_time)) {
        max_query_time = query_interval->time_;
      }

      // add local statistics
      statistics_.add_nested_static_stat("cache",
                                         "local_hit_rates",
                                         'c' + std::to_string(client_id),
                                         chr(received_stats.query_cache_hits, received_stats.query_cache_misses));
      statistics_.add_nested_static_stat(
        "queries", "processed_local", 'c' + std::to_string(client_id), received_stats.processed_queries);
    }

    cn_statistics_.convert(statistics_);
    const auto t_build_max = timing_.create_enroll("build_max");
    const auto t_query_max = timing_.create_enroll("query_max");
    t_build_max->time_ = max_build_time;
    t_query_max->time_ = max_query_time;

    const f64 query_time = t_query_max->get_ms() / 1000.0;  // in sec
    statistics_.add_nested_static_stat(
      "queries", "queries_per_sec", static_cast<u64>(queries_.num_vectors_total / query_time));
    statistics_.add_nested_static_stat(
      "cache", "hit_rate", chr(cn_statistics_.query_cache_hits, cn_statistics_.query_cache_misses));

  } else {
    cm_.initiator_qp->post_send_inlined(&cn_statistics_, sizeof(CNStatistics), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }
}

template <class Distance>
void ComputeNode<Distance>::terminate() {
  constexpr bool done = true;  // dummy value

  // notify memory nodes
  for (const QP& qp : cm_.server_qps) {
    qp->post_send_inlined(&done, sizeof(bool), IBV_WR_SEND);
  }

  context_.poll_send_cq_until_completion(static_cast<i32>(num_servers_));
}

template <class Distance>
f64 ComputeNode<Distance>::compute_local_recall(const ComputeThreads& compute_threads,
                                                u32 k,
                                                size_t processed_queries) {
  u32 true_results = 0;

  for (const auto& thread : compute_threads) {
    for (const auto& [q_id, result] : thread->query_results) {
      for (const node_t hit : result) {
        // we use q_id because ground_truth is always fully read
        for (const node_t positive : ground_truth_.get_components(q_id).subspan(0, k)) {
          if (hit == positive) {
            ++true_results;
            break;
          }
        }
      }
    }
  }

  const f64 recall = true_results / static_cast<f64>(processed_queries) / k;
  return recall;
}

template <class Distance>
void ComputeNode<Distance>::run_http_service(hnsw::HNSW<Distance>& hnsw,
                                             WorkerPool& worker_pool,
                                             u32 num_coroutines,
                                             bool pin_threads,
                                             Configuration& config) {
  print_status("Starting HTTP service mode");
  
  while (http_server_->is_running()) {
    auto task_opt = http_server_->get_next_task();
    if (!task_opt) {
      break;
    }
    
    auto& task = *task_opt;
    if (task.type == http_server::RequestTask::INSERT) {
      process_http_insert(task.insert_req, task.promise, hnsw, worker_pool, num_coroutines, pin_threads);
    } else if (task.type == http_server::RequestTask::QUERY) {
      process_http_query(task.query_req, task.promise, hnsw, worker_pool, num_coroutines, pin_threads);
    } else if (task.type == http_server::RequestTask::SAVE) {
      process_http_save(task.promise, task.save_req, config);
    } else if (task.type == http_server::RequestTask::LOAD) {
      process_http_load(task.promise, task.load_req, config);
    }
  }
  
  print_status("HTTP service stopped");
}

template <class Distance>
void ComputeNode<Distance>::process_http_insert(const http_server::InsertRequest& req,
                                                std::promise<nlohmann::json>& promise,
                                                hnsw::HNSW<Distance>& hnsw,
                                                WorkerPool& worker_pool,
                                                u32 num_coroutines,
                                                bool pin_threads) {
  try {
    node_t id = req.id;
    if (id == static_cast<node_t>(-1)) {
      id = next_http_insert_id_.fetch_add(1);
    }
    
    auto& compute_threads = worker_pool.get_compute_threads();
    u32 num_threads = compute_threads.size();
    u32 worker_id = next_worker_id_.fetch_add(1) % num_threads;
    auto& thread = compute_threads[worker_id];
    
    const u32 coroutine_id = 0;
    
    thread->reset();
    thread->post_balances[coroutine_id] = 0;
    thread->coroutines.clear();
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(hnsw.insert(id, std::span<element_t>(const_cast<element_t*>(req.vector.data()), req.vector.size()), thread)));
    
    for (;;) {
      bool all_done = true;
      thread->poll_cq();
      
      auto& coroutine = *thread->coroutines[coroutine_id];
      if (coroutine.handle.done()) {
      } else if (thread->is_ready(coroutine_id)) {
        all_done = false;
        thread->set_current_coroutine(coroutine_id);
        coroutine.handle.resume();
      } else {
        all_done = false;
      }
      
      if (all_done && coroutine.handle.done()) {
        break;
      }
    }
    
    json response = {
      {"success", true},
      {"id", id},
      {"message", "Vector inserted successfully"}
    };
    http_server_->submit_response(promise, response);
  } catch (const std::exception& e) {
    json response = {
      {"success", false},
      {"error", e.what()}
    };
    http_server_->submit_response(promise, response);
  }
}

template <class Distance>
void ComputeNode<Distance>::process_http_query(const http_server::QueryRequest& req,
                                               std::promise<nlohmann::json>& promise,
                                               hnsw::HNSW<Distance>& hnsw,
                                               WorkerPool& worker_pool,
                                               u32 num_coroutines,
                                               bool pin_threads) {
  try {
    auto& compute_threads = worker_pool.get_compute_threads();
    u32 num_threads = compute_threads.size();
    u32 worker_id = next_worker_id_.fetch_add(1) % num_threads;
    auto& thread = compute_threads[worker_id];
    u32 q_id = 0;
    
    const u32 coroutine_id = 0;
    
    thread->reset();
    thread->post_balances[coroutine_id] = 0;
    thread->query_results.clear();
    thread->coroutines.clear();
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(hnsw.knn(q_id, std::span<element_t>(const_cast<element_t*>(req.vector.data()), req.vector.size()), thread)));
    
    for (;;) {
      bool all_done = true;
      thread->poll_cq();
      
      auto& coroutine = *thread->coroutines[coroutine_id];
      if (coroutine.handle.done()) {
      } else if (thread->is_ready(coroutine_id)) {
        all_done = false;
        thread->set_current_coroutine(coroutine_id);
        coroutine.handle.resume();
      } else {
        all_done = false;
      }
      
      if (all_done && coroutine.handle.done()) {
        break;
      }
    }
    
    std::vector<node_t> results;
    auto it = thread->query_results.find(q_id);
    if (it != thread->query_results.end()) {
      results = it->second;
    }
    
    json response = {
      {"success", true},
      {"results", results},
      {"k", req.k},
      {"ef_search", req.ef_search}
    };
    http_server_->submit_response(promise, response);
  } catch (const std::exception& e) {
    json response = {
      {"success", false},
      {"error", e.what()}
    };
    http_server_->submit_response(promise, response);
  }
}

template <class Distance>
void ComputeNode<Distance>::process_http_save(std::promise<nlohmann::json>& promise, 
                                               const http_server::SaveRequest& req, 
                                               Configuration& config) {
  try {
    if (!cm_.is_initiator) {
      json response = {
        {"success", false},
        {"error", "Only initiator can save index"}
      };
      http_server_->submit_response(promise, response);
      return;
    }

    std::cerr << "[HTTP] Save index requested" << std::endl;
    
    const size_t num_memory_servers = cm_.server_qps.size();

    filepath_t base_path = config.data_path;
    if (!req.path.empty()) {
      base_path = filepath_t(req.path);
    }

    struct Message {
      bool load;
      size_t path_length;
    };

    filepath_t saved_path;

    for (idx_t i = 0; i < num_memory_servers; ++i) {
      filepath_t path = base_path;
      path /= "dump/index_m" + std::to_string(config.m) + "_efc" + std::to_string(config.ef_construction) + "_node" +
              std::to_string(i + 1) + "_of" + std::to_string(num_memory_servers) + ".dat";
      
      if (i == 0) {
        saved_path = path;
      }

      const Message msg{false, path.string().size()};
      const QP& qp = cm_.server_qps[i];

      qp->post_send_inlined(&msg, sizeof(Message), IBV_WR_SEND);
      qp->post_send_inlined(path.string().data(), path.string().size(), IBV_WR_SEND);
    }

    context_.poll_send_cq_until_completion(static_cast<i32>(2 * num_memory_servers));

    const bool success = cm_.synchronize();
    if (!success) {
      throw std::runtime_error("Index saving failed");
    }
    
    json response = {
      {"success", true},
      {"message", "Index saved successfully"},
      {"path", saved_path.string()}
    };
    http_server_->submit_response(promise, response);
  } catch (const std::exception& e) {
    json response = {
      {"success", false},
      {"error", e.what()}
    };
    http_server_->submit_response(promise, response);
  }
}

template <class Distance>
void ComputeNode<Distance>::process_http_load(std::promise<nlohmann::json>& promise, 
                                               const http_server::LoadRequest& req, 
                                               Configuration& config) {
  try {
    if (!cm_.is_initiator) {
      json response = {
        {"success", false},
        {"error", "Only initiator can load index"}
      };
      http_server_->submit_response(promise, response);
      return;
    }

    std::cerr << "[HTTP] Load index requested" << std::endl;
    
    const size_t num_memory_servers = cm_.server_qps.size();

    filepath_t base_path = config.data_path;
    if (!req.path.empty()) {
      base_path = filepath_t(req.path);
    }

    struct Message {
      bool load;
      size_t path_length;
    };

    filepath_t loaded_path;

    for (idx_t i = 0; i < num_memory_servers; ++i) {
      filepath_t path = base_path;
      path /= "dump/index_m" + std::to_string(config.m) + "_efc" + std::to_string(config.ef_construction) + "_node" +
              std::to_string(i + 1) + "_of" + std::to_string(num_memory_servers) + ".dat";
      
      if (i == 0) {
        loaded_path = path;
      }

      const Message msg{true, path.string().size()};
      const QP& qp = cm_.server_qps[i];

      qp->post_send_inlined(&msg, sizeof(Message), IBV_WR_SEND);
      qp->post_send_inlined(path.string().data(), path.string().size(), IBV_WR_SEND);
    }

    context_.poll_send_cq_until_completion(static_cast<i32>(2 * num_memory_servers));

    const bool success = cm_.synchronize();
    if (!success) {
      throw std::runtime_error("Index loading failed");
    }
    
    json response = {
      {"success", true},
      {"message", "Index loaded successfully"},
      {"path", loaded_path.string()}
    };
    http_server_->submit_response(promise, response);
  } catch (const std::exception& e) {
    json response = {
      {"success", false},
      {"error", e.what()}
    };
    http_server_->submit_response(promise, response);
  }
}

// explicitly initiate templates
template ComputeNode<L2Distance>::ComputeNode(Configuration& config);
template ComputeNode<IPDistance>::ComputeNode(Configuration& config);