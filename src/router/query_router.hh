#pragma once

#include <library/hugepage.hh>
#include <library/queue_pair.hh>
#include <random>

#include "cache/placement.hh"
#include "common/constants.hh"
#include "common/statistics.hh"
#include "common/timing.hh"
#include "common/types.hh"
#include "io/database.hh"
#include "message_wrapper.hh"
#include "node/node.hh"

namespace query_router {

template <class Distance>
class QueryRouter {
  using Queries = io::Database<element_t>;

  static constexpr u32 termination_header_mn = static_cast<u32>(-1);
  static constexpr u32 termination_header_cn = static_cast<u32>(-2);
  static constexpr u32 tombstone = static_cast<u32>(-1);

public:
  QueryRouter(const Placement<Distance>& placement,
              Queries& queries,
              Context& context,
              QPs& qps,
              u32 num_compute_nodes,
              i32 max_cqes,
              u32 client_id,
              timing::Timing::IntervalPtr&& timer)
      : message_size(sizeof(u32) + sizeof(node_t) +
                     queries.dim * sizeof(element_t)),  // (header | query id | query components)
        batch_size(LIMIT_PER_CN * num_compute_nodes),
        cn_id_(client_id),
        num_cns_(num_compute_nodes),
        max_cqes_(max_cqes),
        buffer_entries_(batch_size * 2 * (qps.size() + 1)),
        placement_(placement),
        queries_(queries),
        buffer_(buffer_entries_ * message_size),
        qps_(qps),
        lmr_(context, buffer_.get_full_buffer(), buffer_.buffer_size),
        histogram_(num_compute_nodes, 0),
        limits_(num_compute_nodes, LIMIT_PER_CN),
        progresses_(num_compute_nodes, tombstone),
        progresses_ahead_(num_compute_nodes, tombstone),
        send_wcs_(max_cqes),
        recv_wcs_(max_cqes),
        timer_(timer) {
    for (idx_t i = 0; i < buffer_entries_; ++i) {
      freelist_.push_back(i * message_size);
    }

    std::cerr << "routing buffer size: " << buffer_entries_ * message_size << std::endl;
    buffer_.touch_memory();

    // initialize PRNG
    dist_ = std::uniform_int_distribution<u32>(0, qps.size() - 1);
  }

  idx_t pop_freelist() {
    lib_assert(not freelist_.empty(), "empty freelist");

    const idx_t offset = freelist_.back();
    freelist_.pop_back();

    return offset;
  }

  void post_receive(u32 node_id) {
    const idx_t offset = pop_freelist();
    lib_assert(posted_recvs_ < max_cqes_, "?-?-?(1)");

    const u64 wr_id = encode_64bit(node_id, offset);
    qps_[node_id]->post_receive(lmr_, message_size, wr_id, offset);
    ++posted_recvs_;
  }

  void route_query(node_t query_id, const span<element_t> components, u32 destination, u32 router) {
    const idx_t offset = pop_freelist();

    // copy query header to send buffer
    *reinterpret_cast<u32*>(buffer_.get_full_buffer() + offset) = destination;
    *reinterpret_cast<node_t*>(buffer_.get_full_buffer() + offset + sizeof(u32)) = query_id;

    // copy query components to send buffer
    auto* component_ptr =
      reinterpret_cast<element_t*>(buffer_.get_full_buffer() + offset + sizeof(u32) + sizeof(node_t));
    for (idx_t i = 0; i < Node::DIM; ++i) {
      *(component_ptr + i) = components[i];
    }

    // std::cerr << "  ++ route query " << query_id << " to CN " << destination << std::endl;

    lib_assert(posted_sends_ < max_cqes_, "?-?-?(2)");

    const u64 wr_id = encode_64bit(router, offset);
    qps_[router]->post_send_with_id(lmr_, message_size, IBV_WR_SEND, wr_id, true, nullptr, 0, offset);
    ++posted_sends_;
  }

  void update_limits() {
    const f64 sum_progresses = std::accumulate(progresses_.begin(), progresses_.end(), 0u);

    // std::cerr << "update limits wrt received progresses: [";
    // for (const auto p : progresses_) {
    //   std::cerr << p << " ";
    // }
    // std::cerr << "\b]\n";

    if (sum_progresses < num_cns_) {
      std::cerr << "no update\n";
      return;
    }

    // std::cerr << "update from: [";
    // for (const auto limit : limits) {
    //   std::cerr << limit << " ";
    // }
    // std::cerr << "\b]\n";

    f64 denom = 0;
    for (idx_t i = 0; i < num_cns_; ++i) {
      denom += (sum_progresses - progresses_[i]);
    }

    size_t new_batch_size = 0;
    for (idx_t i = 0; i < num_cns_; ++i) {
      const f64 scale_factor = ((sum_progresses - progresses_[i]) / denom) * static_cast<f64>(num_cns_);
      limits_[i] = static_cast<f64>(LIMIT_PER_CN) * scale_factor;

      new_batch_size += limits_[i];
    }

    for (idx_t i = 0; new_batch_size < batch_size; ++i) {
      ++limits_[i % num_cns_];
      ++new_batch_size;
    }

    // std::cerr << "update to: [";
    // for (const auto limit : limits) {
    //   std::cerr << limit << " ";
    // }
    // std::cerr << "\b]\n";

    lib_assert(new_batch_size == batch_size, "invalid batch size");
  }

  bool received_all_acks() const {
    for (idx_t cn = 0; cn < num_cns_; ++cn) {
      if (cn != cn_id_ && progresses_[cn] == tombstone) {
        return false;
      }
    }

    return true;
  }

  size_t poll_recv_cq() {
    const auto& recv_cq = qps_.front()->get_ibv_qp()->recv_cq;
    const u32 num_received = Context::poll_recv_cq(recv_wcs_.data(), max_cqes_, recv_cq);
    posted_recvs_ -= static_cast<i32>(num_received);

    for (u32 i = 0; i < num_received; ++i) {
      const auto [router, offset] = decode_64bit(recv_wcs_[i].wr_id);
      const u32 destination = *reinterpret_cast<u32*>(buffer_.get_full_buffer() + offset);

      if (destination == termination_header_cn) {
        ++received_cn_terminations_;
        freelist_.push_back(offset);
        post_receive(router);

      } else if (destination == termination_header_mn) {
        ++received_mn_terminations_;
        freelist_.push_back(offset);
        // do not post another receive

      } else {  // either received a query or an acknowledgement
        lib_assert(destination == cn_id_, "wrong destination: " + std::to_string(destination));

        AckMessageWrapper ack_msg{buffer_.get_full_buffer() + offset};
        if (ack_msg.header() == AckMessageWrapper::msg_header) {  // received batch acknowledgement
          if (progresses_[ack_msg.sender()] == tombstone) {
            progresses_[ack_msg.sender()] = ack_msg.progress();

          } else {
            lib_assert(progresses_ahead_[ack_msg.sender()] == tombstone, "ahead CHRs already set");
            progresses_ahead_[ack_msg.sender()] = ack_msg.progress();
          }

        } else {
          // copy received message to query buffer
          MessageWrapper msg{buffer_.get_full_buffer() + offset};
          queries_.set_id(queries_.max_slot, msg.query_id());
          auto query_span = queries_.get_components(queries_.max_slot);

          // std::cerr << "  ++ received query " << msg.query_id() << " from MN " << router << std::endl;

          for (idx_t j = 0; j < queries_.dim; ++j) {
            query_span[j] = msg.components()[j];
          }

          ++queue_size;
          query_queue.enqueue(queries_.max_slot);
          ++queries_.max_slot;
        }

        freelist_.push_back(offset);
        post_receive(router);
      }
    }

    return num_received;
  }

  void poll_send_cq() {
    const auto& send_cq = qps_.front()->get_ibv_qp()->send_cq;
    Context::poll_send_cq(send_wcs_.data(), max_cqes_, send_cq, [&](u64 wr_id) {
      auto [_, buffer_offset] = decode_64bit(wr_id);
      freelist_.push_back(buffer_offset);
      --posted_sends_;
    });
  }

  /**
   * @brief Send acknowledgement messages inclusive current progress to each CN via random MNs
   *        MN will not notice (header is just the destination)
   */
  void send_batch_acks_with_progress(u32 progress) {
    progresses_[cn_id_] = progress;

    for (u32 cn_dest = 0; cn_dest < num_cns_; ++cn_dest) {
      if (cn_dest != cn_id_) {
        const idx_t offset = pop_freelist();

        AckMessageWrapper msg{buffer_.get_full_buffer() + offset};
        msg.set(cn_dest, cn_id_, progress);

        const u32 router = get_random_memory_node();
        const u64 wr_id = encode_64bit(router, offset);

        for (idx_t timeout = 0; posted_sends_ >= max_cqes_; ++timeout) {
          lib_assert(timeout < POLL_TIMEOUT, "send batch acks poll timeout");
          poll_send_cq();
        }

        qps_[router]->post_send_with_id(lmr_, message_size, IBV_WR_SEND, wr_id, true, nullptr, 0, offset);
        ++posted_sends_;
      }
    }
  }

  // to all memory nodes (should be called once per thread)
  void send_termination_messages() {
    print_status("send termination message to all MNs");

    for (const QP& qp : qps_) {
      const idx_t offset = pop_freelist();

      // copy query header to send buffer
      *reinterpret_cast<u32*>(buffer_.get_full_buffer() + offset) = termination_header_mn;
      const u64 wr_id = encode_64bit(termination_header_mn, offset);

      while (posted_sends_ >= max_cqes_) {
        poll_send_cq();
      }

      qp->post_send_with_id(lmr_, message_size, IBV_WR_SEND, wr_id, true, nullptr, 0, offset);
      ++posted_sends_;
    }
  }

  bool terminated() const { return received_mn_terminations_ == qps_.size() * INITIAL_RECVS; }
  u32 get_random_memory_node() { return dist_(generator_); }

  void run_routing(u32) {
    print_status("routing thread running...");
    timer_->start();

    const size_t recv_size_per_batch = LIMIT_PER_CN * (num_cns_ - 1);
    lib_assert(static_cast<i32>(recv_size_per_batch) <= max_cqes_, "...");

    // post initial receives for routes
    for (u32 memory_node = 0; memory_node < qps_.size(); ++memory_node) {
      for (idx_t i = 0; i < INITIAL_RECVS; ++i) {
        post_receive(memory_node);
      }
    }

    const idx_t max_local_slot = queries_.max_slot - 1;
    idx_t local_slot = 0;

    for (;;) {
      // limit reached, i.e., a batch is processed, now sync with other CNs
      if (local_slot > 0 && local_slot % batch_size == 0) {
        for (size_t& freq : histogram_) {
          freq = 0;
        }

        const u32 progress = queue_size;
        while (queue_size > MAX_QUEUE_SIZE) {  // sync point
          std::this_thread::sleep_for(std::chrono::milliseconds(100));  // avoid constant polling
        }

        send_batch_acks_with_progress(progress);
        while (not received_all_acks()) {  // acks contain progress (queue size) of other CNs
          poll_recv_cq();
        }
        // std::cerr << "received all acks\n";

        if constexpr (ADAPTIVE_ROUTING) {
          update_limits();  // update batch size of each CN based on their progress
        }

        // reset progresses (can be ahead at most by one)
        for (idx_t j = 0; j < num_cns_; ++j) {
          progresses_[j] = progresses_ahead_[j];
          progresses_ahead_[j] = tombstone;
        }
      }

      // processed all queries, now handle final batch and send termination messages
      if (local_slot > max_local_slot) {
        if (local_slot % batch_size != 0) {  // send final ack messages if not sent yet
          send_batch_acks_with_progress(queue_size);

          while (not received_all_acks()) {
            poll_recv_cq();
          }
        }

        send_termination_messages();

        while (not terminated()) {  // wait until all CNs are done
          poll_recv_cq();
        }

        while (posted_sends_ > 0 || posted_recvs_ > 0) {
          poll_send_cq();
          poll_recv_cq();
        }

        done = true;
        timer_->stop();

        break;
      }

      idx_t destination = 0;

      // determine best matching CN
      auto min_placement = placement_.closest_centroids(queries_.get_components(local_slot));
      while (not min_placement.empty()) {
        destination = min_placement.top().first;
        min_placement.pop();

        if constexpr (BALANCED_ROUTING) {
          if (histogram_[destination] < limits_[destination]) {
            break;
          }
        } else {
          break;
        }
      }

      ++histogram_[destination];

      if (destination == cn_id_) {
        ++queue_size;
        query_queue.enqueue(local_slot);

      } else {
        while (posted_sends_ >= max_cqes_) {
          poll_send_cq();
        }

        route_query(
          queries_.get_id(local_slot), queries_.get_components(local_slot), destination, get_random_memory_node());
      }

      ++local_slot;
    }
  }

public:
  const size_t message_size;
  const size_t batch_size;

  concurrent_queue<idx_t> query_queue;
  std::atomic<i32> queue_size{0};
  std::atomic<bool> done = false;

private:
  const u32 cn_id_;
  const u32 num_cns_;
  const i32 max_cqes_;
  const size_t buffer_entries_;
  const Placement<Distance>& placement_;
  Queries& queries_;

  size_t received_mn_terminations_{0};
  size_t received_cn_terminations_{0};  // we will overall receive CN x MN many

  HugePage<byte_t> buffer_;
  QPs& qps_;
  vec<idx_t> freelist_;
  LocalMemoryRegion lmr_;

  vec<size_t> histogram_;  // track routed queries per CN
  vec<size_t> limits_;  // adjustable limits per CN
  vec<u32> progresses_;  // received progress indicators of other CNs
  vec<u32> progresses_ahead_;  // received progresses of other CNs ahead of this router (can be ahead at most by one)

  vec<ibv_wc> send_wcs_;
  vec<ibv_wc> recv_wcs_;

  i32 posted_sends_{0};
  i32 posted_recvs_{0};

  std::mt19937 generator_{std::random_device{}()};
  std::uniform_int_distribution<u32> dist_;
  timing::Timing::IntervalPtr timer_;
};

}  // namespace query_router
