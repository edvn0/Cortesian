//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/utils/DataSplit.h"

DataSplit::DataSplit(size_t batch_size, const std::vector<Eigen::VectorXd> &Xs,
                     const std::vector<Eigen::VectorXd> &Ys) {
  assert(Xs.size() == Ys.size());
  std::vector<DataSet> batches;
  size_t ds_size = Xs.size();
  size_t num_batches = ds_size / batch_size;

  for (size_t i = 0; i < num_batches; i++) {
    DataSet s;
    size_t fromIndex = i * batch_size;
    size_t toIndex = std::min(ds_size, (i + 1) * batch_size);

    for (; fromIndex < toIndex; fromIndex++) {
      s.rows.push_back(DataPoint{Xs[fromIndex], Ys[fromIndex]});
    }

    batches.emplace_back(s);
  }

  split = batches;
}

std::vector<DataSplit::DataSet> DataSplit::get_splits(bool shuffle) {
  if (shuffle) {
    std::vector<DataSplit::DataSet> to_shuffle = split;
    effolkronium::random_static::shuffle(to_shuffle);
    return to_shuffle;
  } else {
    return split;
  }
}
