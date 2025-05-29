#pragma once

#include <iostream>
#include <library/utils.hh>
#include <random>

#include "hnsw/distance.hh"
#include "node/node.hh"

template <class Distance>
class Kmeans {
  static constexpr u32 ITERATION_LIMIT = 1000;

public:
  using Nodes = vec<s_ptr<Node>>;
  using Centroids = vec<vec<f32>>;

  /**
   * @brief If k is even, run balanced k-means.
   *        Otherwise, run balanced k-means with k=2*k and combine k times the 2 closest clusters (based on centroids).
   *        Balanced k-means produces highly unbalanced clusters if k is small and an odd number.
   *        This is just a heuristic, but works well.
   */
  static std::pair<Centroids, vec<idx_t>> run_and_optimize(const Nodes& nodes, u32 k) {
    print_status("run balanced k-means");
    const u32 local_k = (k % 2 == 0) ? k : 2 * k;

    vec<size_t> new_cluster_sizes(k);
    vec<idx_t> mapping(local_k);

    // normal balanced k-means
    std::cerr << "run kmeans" << std::endl;
    auto [centroids, cluster_assignment, cluster_sizes] = run_kmeans(nodes, local_k);

    std::cerr << "run balanced kmeans" << std::endl;
    const vec<size_t> balanced_cluster_sizes =
      balanced_kmeans(centroids, cluster_assignment, cluster_sizes, nodes, local_k, 0.15, 1.01, 1);

    if (k % 2 == 0) {
      new_cluster_sizes = balanced_cluster_sizes;

      for (idx_t i = 0; i < k; ++i) {
        mapping[i] = i;
      }

    } else {  // combination case: combine 2 closest clusters
      std::cerr << "combine clusters" << std::endl;
      vec<bool> is_assigned(local_k);

      for (idx_t i = 0, next_idx = 0; i < centroids.size(); ++i) {
        if (not is_assigned[i]) {
          distance_t min_dist = std::numeric_limits<distance_t>::max();
          idx_t min_pos = 0;

          for (idx_t j = i + 1; j < centroids.size(); ++j) {
            if (not is_assigned[j]) {
              const distance_t dist = Distance::dist(centroids[i], centroids[j], Node::DIM);

              if (dist < min_dist) {
                min_dist = dist;
                min_pos = j;
              }
            }
          }

          lib_assert(i != min_pos, "invalid assignment");
          is_assigned[i] = true;
          is_assigned[min_pos] = true;

          mapping[i] = next_idx;
          mapping[min_pos] = next_idx;

          new_cluster_sizes[next_idx] = balanced_cluster_sizes[i] + balanced_cluster_sizes[min_pos];
          ++next_idx;
        }
      }
    }

    std::cerr << "final cluster sizes:\n";
    for (idx_t i = 0; i < k; ++i) {
      std::cerr << "  cluster " << i << ": " << new_cluster_sizes[i] << std::endl;
    }

    std::cerr << "final mapping: ";
    for (const auto m : mapping) {
      std::cerr << m << " ";
    }
    std::cerr << std::endl;

    return {centroids, mapping};
  }

  static std::tuple<Centroids, vec<idx_t>, vec<size_t>> run_kmeans(const Nodes& nodes, u32 k) {
    assert(nodes.size() >= k);  // there must be at least k data points

    vec<idx_t> cluster_assignment;
    Centroids centroids;

    f32 error = std::numeric_limits<f32>::max();
    u32 iteration = 0;

    while (iteration < ITERATION_LIMIT && error > 0.001) {
      Centroids new_centroids =
        iteration == 0 ? init_plusplus(nodes, k) : calculate_means(nodes, cluster_assignment, centroids, k);

      cluster_assignment = compute_cluster_assignment(nodes, new_centroids);

      // compute error
      if (iteration > 0) {
        error = 0.;
        for (idx_t i = 0; i < k; ++i) {
          if constexpr (std::is_same_v<Distance, L2Distance>) {
            error += std::sqrt(Distance::dist(centroids[i], new_centroids[i], Node::DIM));
          } else {
            error += Distance::dist(centroids[i], new_centroids[i], Node::DIM);
          }
        }
      }

      centroids = new_centroids;
      ++iteration;
    }

    vec<size_t> cluster_sizes(k);
    for (const idx_t assignment : cluster_assignment) {
      ++cluster_sizes[assignment];
    }

    std::cerr << "cluster sizes:\n";
    for (idx_t i = 0; i < k; ++i) {
      std::cerr << "  cluster " << i << ": " << cluster_sizes[i] << std::endl;
    }

    std::cerr << "  final iteration: " << iteration << ", final error: " << error << std::endl;

    return {centroids, cluster_assignment, cluster_sizes};
  }

private:
  /**
   * @brief Compute for each node in `nodes` the smallest distance to the centroids.
   */
  static vec<f32> closest_distances(const Nodes& node_centroids, const Nodes& nodes) {
    vec<f32> distances;
    distances.reserve(nodes.size());

    for (auto& node : nodes) {
      f32 closest = std::numeric_limits<f32>::max();

      for (auto& centroid : node_centroids) {
        const f32 distance = Distance::dist(node->components(), centroid->components(), Node::DIM);
        if (distance < closest) {
          closest = distance;
        }
      }

      distances.push_back(closest);
    }

    return distances;
  }

  static Centroids init_plusplus(const Nodes& nodes, u32 k) {
    Nodes node_centroids;
    node_centroids.reserve(k);

    // select first centroid at random from `nodes`
    {
      std::mt19937 generator{1234};  // NOLINT (fixed seed to ensure consistency across the compute nodes)
      auto dist = std::uniform_int_distribution<idx_t>(0, nodes.size() - 1);
      node_centroids.push_back(nodes[dist(generator)]);
    }

    // select remaining centroids
    while (node_centroids.size() < k) {
      // calculate for each node the smallest distance to the centroids
      auto distances = closest_distances(node_centroids, nodes);

      // pick node where the minimum distance to the previous centroids is maximum as new centroid
      const auto max_iter = std::max_element(distances.begin(), distances.end());
      const idx_t max_pos = std::distance(distances.begin(), max_iter);

      node_centroids.push_back(nodes[max_pos]);
    }

    // copy components from `node_centroids` to `centroids` (since we will update the dimensions)
    Centroids centroids(k, vec<f32>(Node::DIM));

    // copy components
    for (idx_t i = 0; i < k; ++i) {
      for (idx_t j = 0; j < Node::DIM; ++j) {
        centroids[i][j] = node_centroids[i]->components()[j];
      }
    }

    return centroids;
  }

  /**
   * @brief Compute a cluster assignment (a cluster index \in [0, k-1]) for each node.
   */
  static vec<idx_t> compute_cluster_assignment(const Nodes& nodes, const Centroids& centroids) {
    vec<idx_t> cluster_assignment;

    for (const auto& node : nodes) {
      f32 smallest_distance = std::numeric_limits<f32>::max();
      idx_t index = 0;

      // determine closest centroid
      for (idx_t i = 0; i < centroids.size(); ++i) {
        const f32 dist = Distance::dist(node->components(), centroids[i], Node::DIM);

        if (dist < smallest_distance) {
          smallest_distance = dist;
          index = i;
        }
      }

      cluster_assignment.push_back(index);
    }

    return cluster_assignment;
  }

  static Centroids calculate_means(const Nodes& nodes,
                                   const vec<idx_t>& cluster_assignment,
                                   const Centroids& old_centroids,
                                   u32 k) {
    Centroids new_centroids(k, vec<f32>(Node::DIM));
    vec<f32> count(k, 0.);

    for (idx_t i = 0; i < nodes.size(); ++i) {
      auto& new_centroid = new_centroids[cluster_assignment[i]];
      ++count[cluster_assignment[i]];

      for (idx_t j = 0; j < Node::DIM; ++j) {
        new_centroid[j] += nodes[i]->components()[j];
      }
    }

    for (idx_t i = 0; i < k; ++i) {
      if (count[i] == 0) {
        new_centroids[i] = old_centroids[i];
      } else {
        for (idx_t j = 0; j < Node::DIM; ++j) {
          new_centroids[i][j] /= count[i];
        }
      }
    }

    return new_centroids;
  }

  /**
   * @brief Follows Algorithm 1 from Maeyer et al.'s "Balanced k-means revisited".
   *        https://github.com/uef-machine-learning/Balanced_k-Means_Revisited/
   * @param c Partly remaining fraction
   */
  static vec<size_t> balanced_kmeans(Centroids& centroids,
                                     vec<idx_t>& cluster_assignment,
                                     vec<size_t>& cluster_sizes,
                                     const Nodes& nodes,
                                     u32 k,
                                     f32 c,
                                     f32 penalty_factor,
                                     u32 max_cluster_size_difference) {
    f32 p_now = 0., p_next = std::numeric_limits<f32>::max();
    size_t n_min = 0, n_max = nodes.size();
    idx_t iter = 0;

    // store the sums of the points per cluster for centroid re-computation
    vec sum_coords(k, vec<f32>(Node::DIM, 0.));
    for (idx_t i = 0; i < nodes.size(); ++i) {
      for (idx_t j = 0; j < Node::DIM; ++j) {
        sum_coords[cluster_assignment[i]][j] += nodes[i]->components()[j];
      }
    }

    while (n_max - n_min > max_cluster_size_difference && iter < ITERATION_LIMIT) {  // termination criterion
      for (idx_t node_idx = 0; node_idx < nodes.size(); ++node_idx) {
        const auto& node = nodes[node_idx];
        const idx_t old_cluster_assignment = cluster_assignment[node_idx];

        // ensure that cluster size is > 0
        if (cluster_sizes[old_cluster_assignment] == 1) {
          continue;
        }

        // remove point
        auto& old_centroid = centroids[old_cluster_assignment];
        for (idx_t j = 0; j < Node::DIM; ++j) {
          sum_coords[old_cluster_assignment][j] -= node->components()[j];
          old_centroid[j] =
            sum_coords[old_cluster_assignment][j] / static_cast<f32>(cluster_sizes[old_cluster_assignment] - 1);
        }

        --cluster_sizes[old_cluster_assignment];

        // determine new cluster assignment with penalty
        // penalty computation taken from: https://github.com/uef-machine-learning/Balanced_k-Means_Revisited/
        auto& new_cluster_assignment = cluster_assignment[node_idx];
        f32 cost = std::numeric_limits<f32>::max();

        const f32 dist_old_centroid = Distance::dist(old_centroid, node->components(), Node::DIM);
        const f32 old_size = static_cast<f32>(cluster_sizes[old_cluster_assignment]) + c;

        for (idx_t j = 0; j < k; j++) {
          const f32 dist_j = Distance::dist(centroids[j], node->components(), Node::DIM);
          const f32 size_j = static_cast<f32>(cluster_sizes[j]);
          const f32 penalty_needed = (dist_j - dist_old_centroid) / (old_size - size_j);

          if (old_size > size_j) {
            if (p_now < penalty_needed) {
              if (penalty_needed < p_next) {
                p_next = penalty_needed;
              }
            } else {
              if (dist_j + p_now * size_j < cost && j != old_cluster_assignment) {
                cost = dist_j + p_now * size_j;
                new_cluster_assignment = j;
              }
            }
          } else {
            if (p_now < penalty_needed && dist_j + p_now * size_j < cost) {
              cost = dist_j + p_now * size_j;
              new_cluster_assignment = j;
            }
          }
        }

        // add point to new cluster
        auto& new_centroid = centroids[new_cluster_assignment];
        size_t& new_cluster_size = cluster_sizes[new_cluster_assignment];

        for (idx_t j = 0; j < Node::DIM; ++j) {
          sum_coords[new_cluster_assignment][j] += node->components()[j];
          new_centroid[j] = sum_coords[new_cluster_assignment][j] / static_cast<f32>(new_cluster_size + 1);
        }
        ++new_cluster_size;
      }

      n_min = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
      n_max = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());

      p_now = penalty_factor * p_next;
      p_next = std::numeric_limits<f32>::max();
      ++iter;
    }

    std::cerr << "num iterations: " << iter << std::endl;
    std::cerr << "reported cluster sizes:\n";
    for (idx_t i = 0; i < k; ++i) {
      std::cerr << "  cluster " << i << ": " << cluster_sizes[i] << std::endl;
    }

    vec<size_t> actual_sizes(k);
    for (auto& node : nodes) {
      f32 min_dist = std::numeric_limits<f32>::max();
      idx_t index = 0;
      for (idx_t i = 0; i < k; ++i) {
        const f32 dist = Distance::dist(centroids[i], node->components(), Node::DIM);
        if (dist < min_dist) {
          min_dist = dist;
          index = i;
        }
      }

      actual_sizes[index]++;
    }

    std::cerr << "actual cluster sizes:\n";
    for (idx_t i = 0; i < k; ++i) {
      std::cerr << "  cluster " << i << ": " << actual_sizes[i] << std::endl;
    }

    return actual_sizes;
  }
};
