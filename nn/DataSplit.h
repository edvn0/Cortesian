//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_DATASPLIT_H
#define CORTESIAN_DATASPLIT_H

#include <Eigen/Core>
#include <effolkronium/random.hpp>
#include <vector>

class DataSplit {
 public:
  struct DataPoint {
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
  };

 private:
  struct DataSet {
    std::vector<DataPoint> rows;
  };

  std::vector<DataSet> split;

 public:
  DataSplit(size_t batch_size, const std::vector<Eigen::VectorXd> &Xs,
            const std::vector<Eigen::VectorXd> &Ys);

  std::vector<DataSet> get_splits(bool shuffle=false);
};

#endif  // CORTESIAN_DATASPLIT_H
