#pragma once

#include <filesystem>
#include <library/connection_manager.hh>
#include <library/detached_qp.hh>
#include <library/hugepage.hh>
#include <library/utils.hh>

#include "common/configuration.hh"
#include "common/constants.hh"
#include "common/core_assignment.hh"
#include "common/timing.hh"

/**
 *  Memory layout:
 *  -----------------------------
 *    buffer: [ free-ptr(8) | entry-ptr(8) | node_a | node_b | ... ]
 *  -----------------------------
 *  Node layout: [
 *     header: 8B                           | ... | ... | is_entry_node(1b) | ... | new_lvl_lock(1b) | ... | lock(1b) |
 *                                                  ^--------- 1B ---------^ ^--------- 1B ---------^ ^----- 1B -----^
 *     meta: 2 * 4B                         | uid(4) | level(4) |
 *     components: d * 4B                   | d_1(4) | ... | d_d(4) |
 *     base-layer: 4B + M_max_0 * 8B        | #neighbors(4) | l_0_1(8) | ... | l_0_M(8) |
 *     upper layer(s) l * (4B + M_max * 8B) | ... |                                        <- only if node's level > 0
 *   ]
 */

/**
 * @brief Establishes a connection to all involved compute nodes.
 *        Allocates a huge memory block and forwards access tokens.
 *        Creates a QP per compute thread and connects them.
 *        Waits until a termination signal is received.
 */
class MemoryNode {
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;

public:
  explicit MemoryNode(Configuration& config)
      : context_(config), cm_(context_, config), num_clients_(config.num_clients), index_region_(context_) {
    cm_.connect_to_clients();

    if (!config.disable_thread_pinning) {
      const u32 core = core_assignment_.get_available_core();
      pin_main_thread(core);
      print_status("pinned main thread to core " + std::to_string(core));
    }

    // receive runtimes parameters from initiator
    configuration::Parameters p{};
    LocalMemoryRegion region{context_, &p, sizeof(configuration::Parameters)};

    cm_.initiator_qp->post_receive(region);
    context_.receive();

    num_compute_threads_ = p.num_threads;
    allocate_memory();

    // free-ptr is initialized to 16 (points to first free address in the buffer)
    *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;

    print_status("register memory and distribute access token");
    index_region_.register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);
    MemoryRegionToken token = index_region_.createToken();

    // send access token to all compute nodes
    for (QP& qp : cm_.client_qps) {
      qp->post_send_inlined(std::addressof(token), sizeof(token), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }

    // connect for each compute thread a new QP
    print_status("connect QPs of compute threads");
    vec<u_ptr<DetachedQP>> qps;

    // note: no need for QP sharing on the memory server side
    const u32 qps_per_node = std::min<u32>(num_compute_threads_, MAX_QPS);
    qps.reserve(num_clients_ * qps_per_node);

    for (QP& client_qp : cm_.client_qps) {
      for (u32 thread_id = 0; thread_id < qps_per_node; ++thread_id) {
        auto& qp = qps.emplace_back(std::make_unique<DetachedQP>(context_));
        qp->connect(context_, context_.get_lid(), client_qp);
      }
    }

    // notify compute nodes that we are ready
    cm_.synchronize();

    store_or_load_index();
    if (p.use_cache && cache::CACHE_WARMUP) {
      print_status("cache warmup");

      if (config.num_clients > 1 && p.routing) {
        route_queries(config.max_recv_queue_wr);
      }
      idle();  // wait until we get notifications from all compute nodes to terminate
    }

    // actual queries
    print_status("query processing");
    if (config.num_clients > 1 && p.routing) {
      route_queries(config.max_recv_queue_wr);
    }
    idle();  // wait until we get notifications from all compute nodes to terminate

    std::cout << timing_ << std::endl;
  }

private:
  void allocate_memory() {
    const auto t_allocate = timing_.create_enroll("allocate_index_buffer");
    std::cerr << "allocation size: " << MEMORY_NODE_MAX_MEMORY << std::endl;

    t_allocate->start();
    const size_t available_memory = index_buffer_.get_memory_size();
    lib_assert(MEMORY_NODE_MAX_MEMORY <= available_memory, "allocation failed");

    // pre-allocate all available huge pages
    //      index_buffer_.allocate(index_buffer_.get_memory_size());
    index_buffer_.allocate(MEMORY_NODE_MAX_MEMORY);
    index_buffer_.touch_memory();
    t_allocate->stop();
  }

  /**
   * @brief Receive from initator whether to load or store the index (or do nothing).
   */
  void store_or_load_index() {
    print_status("idle: index construction");

    struct Message {
      bool load;
      size_t path_length;
    } msg{};

    const auto send_error = [&]() {
      constexpr bool ready = false;

      cm_.initiator_qp->post_send_inlined(&ready, sizeof(bool), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    };

    LocalMemoryRegion region{context_, &msg, sizeof(Message)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();

    if (msg.path_length > 0) {  // load or store case
      str path;
      path.resize(msg.path_length);

      LocalMemoryRegion path_region{context_, path.data(), msg.path_length};
      cm_.initiator_qp->post_receive(path_region);
      context_.receive();

      if (msg.load) {  // load index
        std::ifstream file{path, std::ios::binary};

        if (!file.good()) {
          send_error();
          lib_failure("file \"" + path + "\" does not exist");
        }

        file.unsetf(std::ios::skipws);  // without that, data is missing
        size_t file_size;

        // determine file size and set cursor back to the beginning
        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size > index_buffer_.buffer_size) {
          send_error();
          lib_failure("cannot load index (buffer too small)");
        }

        print_status("loading index (" + std::to_string(file_size) + " Bytes) from " + path);
        auto t_read = timing_.create_enroll("read_index_buffer");

        t_read->start();
        file.read(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), file_size);
        t_read->stop();

      } else {  // store index
        const auto t_store = timing_.create_enroll("store_index_buffer");
        const size_t index_size = *reinterpret_cast<u64*>(index_buffer_.get_full_buffer());

        print_status("storing index (" + std::to_string(index_size) + " Bytes) to " + path);

        create_directory(filepath_t{path}.parent_path());  // create directory if not exists
        std::ofstream output_s{path, std::ios::out | std::ios::binary};

        t_store->start();
        if (!output_s.write(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), index_size)) {
          lib_failure("Cannot write to file");
        }
        t_store->stop();

        output_s.close();
      }
    }

    // notify initiator that we are done
    constexpr bool ready = true;

    cm_.initiator_qp->post_send_inlined(&ready, sizeof(bool), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }

  void route_queries(i32 max_cqes) {
    print_status("route queries");
    size_t num_routings = 0;

    // receive routing message size
    size_t message_size;
    {
      LocalMemoryRegion region{context_, &message_size, sizeof(message_size)};
      cm_.initiator_qp->post_receive(region);
      context_.receive();

      std::cerr << "routing message size: " << message_size << " B\n";
    }

    const size_t buffer_entries = num_clients_ * query_router::LIMIT_PER_CN * (num_clients_ - 1) * 2;

    HugePage<byte_t> routing_buffer(buffer_entries * message_size);
    routing_buffer.touch_memory();

    LocalMemoryRegion lmr{
      context_, routing_buffer.get_full_buffer(), routing_buffer.buffer_size};  // register memory region

    vec<idx_t> freelist;  // offsets
    freelist.reserve(buffer_entries);

    for (idx_t i = 0; i < buffer_entries; ++i) {
      freelist.push_back(i * message_size);
    }

    constexpr u32 termination_signal_mn = static_cast<u32>(-1);
    constexpr u32 termination_signal_cn = static_cast<u32>(-2);
    u32 received_termination_signals = 0;

    vec<ibv_wc> recv_wcs(max_cqes);
    vec<ibv_wc> send_wcs(max_cqes);

    i32 posted_sends = 0;
    i32 posted_recvs = 0;

    cm_.synchronize();  // synchronize with CNs

    const auto post_receive = [&](u32 client) {
      lib_assert(!freelist.empty(), "empty freelist");
      const idx_t offset = freelist.back();
      freelist.pop_back();

      lib_assert(posted_recvs < max_cqes, "?-?-?(3)");

      const u64 wr_id = encode_64bit(client, offset);
      cm_.client_qps[client]->post_receive(lmr, message_size, wr_id, offset);
      ++posted_recvs;
    };

    const auto poll_send_cq = [&]() {
      Context::poll_send_cq(send_wcs.data(), max_cqes, context_.get_send_cq(), [&](u64 wr_id) {
        const auto [_, offset] = decode_64bit(wr_id);
        freelist.push_back(offset);
        --posted_sends;
      });
    };

    // post initial receives
    for (u32 client = 0; client < num_clients_; ++client) {
      post_receive(client);
    }

    while (received_termination_signals < num_clients_) {
      // poll for receive completion events: route query
      const u32 num_received = context_.poll_recv_cq(recv_wcs.data(), max_cqes);
      posted_recvs -= static_cast<i32>(num_received);

      for (u32 i = 0; i < num_received; ++i) {
        const auto [client, offset] = decode_64bit(recv_wcs[i].wr_id);
        const u32 destination = *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset);

        if (destination == termination_signal_mn) {
          std::cerr << "received termination signal from CN" << client << std::endl;
          ++received_termination_signals;

          for (idx_t cn_id = 0; cn_id < num_clients_; ++cn_id) {
            if (client != cn_id) {
              lib_assert(!freelist.empty(), "empty freelist");
              const idx_t offset_term = freelist.back();
              freelist.pop_back();
              *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset_term) = termination_signal_cn;

              lib_assert(posted_sends < max_cqes, "?-?-?(1)");

              cm_.client_qps[cn_id]->post_send_with_id(
                lmr, message_size, IBV_WR_SEND, encode_64bit(cn_id, offset_term), true, nullptr, 0, offset_term);
              ++posted_sends;
              std::cerr << " send termination message to CN" << cn_id << std::endl;
            }
          }

          freelist.push_back(offset);

        } else {
          // std::cerr << "route query " << *reinterpret_cast<node_t*>(routing_buffer.data() + offset + sizeof(u32))
          //           << " from CN" << client << " to CN" << destination << std::endl;
          lib_assert(destination < num_clients_, "invalid route " + std::to_string(destination));
          lib_assert(client != destination, "invalid route (client == destination)");

          // possibly unable to send, because receiver side hasn't taken the request yet
          do {
            poll_send_cq();
          } while (posted_sends >= max_cqes);

          lib_assert(posted_sends < max_cqes, "too many posts...");  // TODO: remove
          cm_.client_qps[destination]->post_send_with_id(
            lmr, message_size, IBV_WR_SEND, encode_64bit(destination, offset), true, nullptr, 0, offset);
          ++posted_sends;
          ++num_routings;

          lib_assert(posted_recvs < max_cqes, "too many recv posts...");  // TODO: remove
          post_receive(client);
        }
      }

      poll_send_cq();  // poll for send completion events and push offset(s) back to freelist
    }

    // poll remaining send completion events
    while (posted_sends > 0) {
      poll_send_cq();
    }

    lib_assert(posted_recvs == 0, "uncompleted posted receives");
    lib_assert(posted_sends == 0, "uncompleted posted sends");
    print_status("received all termination messages");

    // finally, send termination message to all CNs
    {
      const idx_t offset = freelist.back();
      freelist.pop_back();
      *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset) = termination_signal_mn;

      for (idx_t cn_id = 0; cn_id < num_clients_; ++cn_id) {
        std::cerr << "send final termination messages to CN" << cn_id << std::endl;
        for (idx_t b = 0; b < query_router::INITIAL_RECVS; ++b) {
          lib_assert(posted_sends < max_cqes, "?-?-?(2)");
          cm_.client_qps[cn_id]->post_send(lmr, message_size, IBV_WR_SEND, true, nullptr, 0, offset);
        }
      }

      context_.poll_send_cq_until_completion(num_clients_ * query_router::INITIAL_RECVS);
      freelist.push_back(offset);
    }

    print_status("done with routing (num routings: " + std::to_string(num_routings) + ')');
    lib_assert(freelist.size() == buffer_entries, "unfreed messages in buffer");
  }

  void idle() {
    print_status("idle: queries");

    // dummy region
    bool done;
    LocalMemoryRegion region{context_, &done, sizeof(bool)};

    for (const QP& qp : cm_.client_qps) {
      qp->post_receive(region);
    }

    // wait
    context_.receive(num_clients_);
  }

private:
  Context context_;
  ServerConnectionManager cm_;
  Assignment core_assignment_;

  const u32 num_clients_;
  u32 num_compute_threads_{};

  HugePage<byte_t> index_buffer_;
  MemoryRegion index_region_;
  timing::Timing timing_;
};
