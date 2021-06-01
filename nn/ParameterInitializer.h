//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_PARAMETERINITIALIZER_H
#define CORTESIAN_PARAMETERINITIALIZER_H

#include "../libs/Eigen/Core"
#include <vector>

class ParameterInitializer {
protected:
  bool m_is_initialized{false};
  std::vector<int> m_structure;
  size_t m_offset_length;

  [[nodiscard]] bool pre_check_get_params() const {
    return m_is_initialized && !m_structure.empty();
  }

public:
  virtual ~ParameterInitializer() = default;

  virtual std::vector<Eigen::MatrixXd> get_weight_params() = 0;
  virtual std::vector<Eigen::VectorXd> get_bias_params() = 0;
  virtual std::vector<Eigen::MatrixXd> get_delta_weight_params() = 0;
  virtual std::vector<Eigen::VectorXd> get_delta_bias_params() = 0;
  virtual void init(std::vector<int> structure) = 0;
};

#endif // CORTESIAN_PARAMETERINITIALIZER_H
