//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_MATHUTILS_H
#define CORTESIAN_MATHUTILS_H

#include "../libs/Eigen/Core"

/**
 * Returns the argument of the maximum of the vector.
 * @param in vector
 * @return index of max
 */
static int arg_max(const Eigen::VectorXd &in) {
  size_t rows = in.rows();
  int max = 0;
  double val = in(0);

  for (int i = 1; i < rows; i++) {
    auto val_at_i = in(i);
    if (val_at_i > val) {
      max = i;
      val = val_at_i;
    }
  }

  return max;
}

/**
 * Returns the max of the vector.
 * @param in vector
 * @return double max
 */
static double max(const Eigen::VectorXd &in) {
  size_t rows = in.rows();
  double val = in(0);

  for (int i = 1; i < rows; i++) {
    auto val_at_i = in(i);
    if (val_at_i > val) {
      val = val_at_i;
    }
  }

  return val;
}

#endif // CORTESIAN_MATHUTILS_H
