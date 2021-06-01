//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_BACKPROPSTATISTICS_H
#define CORTESIAN_BACKPROPSTATISTICS_H

#include <ostream>
#include <vector>

class BackPropStatistics {
private:
  double mean_time_epochs{0.0};
  double mean_loss_epochs{0.0};
  std::vector<double> mean_for_eval_functions;
  size_t iterations{0};

public:
  explicit BackPropStatistics(size_t eval_functions) {
    mean_for_eval_functions.reserve(eval_functions);
    for (size_t i = 0; i < eval_functions; i++) {
      mean_for_eval_functions.emplace_back(0.0);
    }
  }

  void update(double time_epoch, double loss,
              const std::vector<double> &accuracy) {
    for (int i = 0; i < mean_for_eval_functions.size(); i++) {
      mean_for_eval_functions[i] += accuracy[i];
    }
    mean_loss_epochs += loss;
    mean_time_epochs += time_epoch;
    iterations++;
  }

  void finalize() {
    mean_time_epochs /= (double)iterations;
    mean_loss_epochs /= (double)iterations;
    for (double &i : mean_for_eval_functions) {
      i /= (double)iterations;
    }
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const BackPropStatistics &statistics) {
    os << "Mean Time/epoch: " << statistics.mean_time_epochs
       << "\nMean_Loss/epoch: " << statistics.mean_loss_epochs
       << "\nIterations: " << statistics.iterations
       << "\nMean for Evaluation Function(s): [";
    for (const double &i : statistics.mean_for_eval_functions) {
      os << i << ", ";
    }
    os << "]\n";
    return os;
  }
};

#endif // CORTESIAN_BACKPROPSTATISTICS_H
