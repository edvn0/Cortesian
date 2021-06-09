//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_DATASPLIT_H
#define CORTESIAN_DATASPLIT_H

#include <effolkronium/random.hpp>
#include <eigen3/Eigen/Core>
#include <vector>

#include <algorithm>
#include <iterator>
#include <vector>

template <typename Vector>
static auto split_vector(const Vector &v, unsigned number_lines) {
  using Iterator = typename Vector::const_iterator;
  std::vector<Vector> rtn;
  Iterator it = v.cbegin();
  const Iterator end = v.cend();

  while (it != end) {
    Vector v;
    std::back_insert_iterator<Vector> inserter(v);
    const auto num_to_copy =
        std::min(static_cast<unsigned>(std::distance(it, end)), number_lines);
    std::copy(it, it + num_to_copy, inserter);
    rtn.push_back(std::move(v));
    std::advance(it, num_to_copy);
  }

  return rtn;
}

class DataSplit {
public:
  struct DataPoint {
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
  };

  struct DataSet {
    std::vector<DataPoint> rows;
  };

private:
  std::vector<DataSet> split;

public:
  DataSplit(size_t batch_size, const std::vector<Eigen::VectorXd> &Xs,
            const std::vector<Eigen::VectorXd> &Ys);

  std::vector<DataSet> get_splits(bool shuffle = false);
};

#endif // CORTESIAN_DATASPLIT_H
