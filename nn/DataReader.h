//
// Created by Edwin Carlsson on 2021-06-02.
//

#ifndef CORTESIAN_DATAREADER_H
#define CORTESIAN_DATAREADER_H

#include "../libs/csv-parser/single_include/csv.hpp"
#include "BlockTimer.h"
#include <Eigen/Core>
#include <functional>
#include <string>
#include <tuple>
#include <utility>

/**
 * Helper method to load CSV data via functors provided.
 * These functors are important: you are provided with a row and a column index,
 * to be able to provide where in your data is X, Y mapped.
 * @param file_name path of file
 * @param rows how many rows of CSV?
 * @param X_cols after mapping, how many rows does your X tensor have?
 * @param Y_cols after mapping, how many rows does your Y tensor have?
 * @param X_mapper mapping a CSVField to your X data.
 * @param Y_mapper mapping a CSVField to your Y data.
 * @param has_header does this CSV have a header?
 * @param delimiter what delimiter is used?
 * @return a rank (1,1) tensor
 */
static std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
csv_to_tensor(const std::string &file_name, size_t rows, size_t X_cols,
              size_t Y_cols,
              const std::function<void(Eigen::MatrixXd &, csv::CSVField &,
                                       size_t, size_t)> &X_mapper,
              const std::function<void(Eigen::MatrixXd &, csv::CSVField &,
                                       size_t, size_t)> &Y_mapper,
              bool has_header = true, char delimiter = ',') {

  BlockTimer t;
  csv::CSVFormat format;
  format.delimiter(delimiter);
  if (!has_header)
    format.no_header();

  csv::CSVReader reader(file_name, format);
  Eigen::MatrixXd X_tensor;
  X_tensor.resize((long)rows, (long)X_cols);
  Eigen::MatrixXd Y_tensor;
  Y_tensor.resize((long)rows, (long)Y_cols);

  size_t row = 0;
  for (csv::CSVRow &csv_row : reader) {
    size_t column = 0;
    for (csv::CSVField &field : csv_row) {
      X_mapper(X_tensor, field, row, column);
      Y_mapper(X_tensor, field, row, column);
      column++;
    }
    row++;
  }
  auto stopped = t.elapsedSeconds();
  std::cout << "Data loading and mapping took: " << stopped << " seconds.\n";

  return std::make_tuple(X_tensor, Y_tensor);
}

#endif // CORTESIAN_DATAREADER_H
