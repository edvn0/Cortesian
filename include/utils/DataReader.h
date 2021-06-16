//
// Created by Edwin Carlsson on 2021-06-02.
//

#ifndef CORTESIAN_DATAREADER_H
#define CORTESIAN_DATAREADER_H

#include "../../libs/csv-parser/single_include/csv.hpp"
#include "BlockTimer.h"
#include <eigen3/Eigen/Core>
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
static std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> csv_to_tensor(
    const std::string &file_name, size_t rows, size_t X_cols, size_t Y_cols,
    const std::function<void(Eigen::MatrixXd &, csv::CSVRow &)> &X_mapper,
    const std::function<void(Eigen::MatrixXd &, csv::CSVRow &)> &Y_mapper,
    bool has_header = true, char delimiter = ',') {

  BlockTimer t;
  csv::CSVFormat format;
  format.delimiter(delimiter);
  if (!has_header)
    format.no_header();
  else {
    format.header_row(0);
  }

  csv::CSVReader reader(file_name, format);
  Eigen::MatrixXd X_tensor;
  X_tensor.resize((long)rows, (long)X_cols);
  Eigen::MatrixXd Y_tensor;
  Y_tensor.resize((long)rows, (long)Y_cols);

  for (csv::CSVRow &csv_row : reader) {
    X_mapper(X_tensor, csv_row);
    Y_mapper(Y_tensor, csv_row);
  }
  auto stopped = t.elapsedSeconds();
  std::cout << "Data loading and mapping took: " << stopped << " seconds.\n";

  return std::make_tuple(X_tensor, Y_tensor);
}

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
csv_to_mnist(const std::string &file_name, size_t X_cols, size_t Y_cols,
             size_t max, bool has_header = true, char delimiter = ',') {

  BlockTimer t;
  csv::CSVFormat format;
  format.delimiter(delimiter);
  if (!has_header)
    format.no_header();
  else {
    format.header_row(0);
  }

  csv::CSVReader reader(file_name, format);
  Eigen::MatrixXd X_tensor;
  X_tensor.resize((long)max, (long)X_cols);
  Eigen::MatrixXd Y_tensor;
  Y_tensor.resize((long)max, (long)Y_cols);

  size_t row = 0;
  for (csv::CSVRow &csv_row : reader) {
    size_t column = 0;
    for (csv::CSVField &field : csv_row) {
      if (column == 0) {
        Y_tensor(row, field.get<long>()) = 1.0;
      } else {
        X_tensor(row, column - 1) = field.get<double>() / 255.0;
      }
      column++;
    }
    row++;
    if (row >= max) {

      auto stopped = t.elapsedSeconds();
      std::cout << "Data loading and mapping took: " << stopped
                << " seconds.\n";

      return std::make_tuple(X_tensor, Y_tensor);
    }
  }
  auto stopped = t.elapsedSeconds();
  std::cout << "Data loading and mapping took: " << stopped << " seconds.\n";

  return std::make_tuple(X_tensor, Y_tensor);
}

enum class EigenType { VECTOR = 0, MATRIX = 1 };

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
  std::ostringstream out;
  out.precision(n);
  out << a_value;
  return out.str();
}

static const std::string json_to_eigen_bias(const std::string &basicString);
static const std::string json_to_eigen_weight(const std::string &basicString);

template <typename EigenBaseType = Eigen::MatrixXd>
static inline const EigenBaseType
json_to_eigen_bias(std::string &bias_str_no_parentheses, size_t rows);

template <typename EigenBaseType = Eigen::MatrixXd>
static inline const EigenBaseType
json_to_eigen_weight(const std::string &weight_str_no_parentheses, size_t rows,
                     size_t cols);

template <typename EigenBaseType = Eigen::VectorXd>
static inline const std::vector<double>
json_to_eigen_row(std::string &bias_str_no_parentheses, size_t rows);

/**
 * Converts a Eigen Matrix to a nested json array.
 * @param weight matrix
 * @return json nested array.
 */
static inline std::string eigen_to_json(const Eigen::MatrixXd &weight) {
  size_t rows = weight.rows(), cols = weight.cols();
  std::string nested_array = "[";
  nested_array.reserve((rows * cols) * 4);
  for (size_t i = 0; i < rows; i++) {
    std::string nested = "[";
    for (size_t j = 0; j < cols; j++) {
      nested.append(to_string_with_precision(weight(i, j), 20));
      if (j != cols - 1) {
        nested.append(",");
      }
    }
    nested.append("]");
    nested_array.append(nested);
    if (i != rows - 1) {
      nested_array.append(",");
    }
  }
  nested_array.append("]");

  return nested_array;
}

/**
 * Converts a Eigen Vector to a nested json array.
 * @param weight matrix
 * @return json nested array.
 */
static inline std::string eigen_to_json(const Eigen::VectorXd &bias) {
  size_t rows = bias.rows();
  std::string nested_array = "[";
  nested_array.reserve((rows)*4);
  for (size_t i = 0; i < rows; i++) {
    nested_array.append(to_string_with_precision(bias(i), 20));
    if (i != rows - 1) {
      nested_array.append(",");
    }
  }
  nested_array.append("]");

  return nested_array;
}

static inline void replace_in_place(std::string &subject,
                                    const std::string &search,
                                    const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

template <typename EigenBaseType>
static inline const EigenBaseType
json_to_eigen_weight(std::string &weight_str_no_parentheses, size_t rows,
                     size_t cols) {
  EigenBaseType mat(rows, cols);

  std::vector<Eigen::VectorXd> row_vectors;
  // remove the parentheses on either side of each sub array, pass to bias.
  auto first_paren = 0;
  auto next_paren = 0;

  // This is soooo hacky. I have no idea of how to fix this.
  while ((next_paren = weight_str_no_parentheses.find("]")) !=
             std::string::npos &&
         (first_paren = weight_str_no_parentheses.find("[")) !=
             std::string::npos) {
    auto both_paren_gone =
        weight_str_no_parentheses.substr(first_paren + 1, next_paren - 1);
    row_vectors.emplace_back(
        json_to_eigen_bias<Eigen::VectorXd>(both_paren_gone, cols));
    weight_str_no_parentheses.erase(0, next_paren + first_paren + 2);
  }

  for (size_t i = 0; i < rows; i++) {
    auto vec = row_vectors[i];
    for (size_t j = 0; j < cols; j++) {
      mat(i, j) = vec(j);
    }
  }

  return mat;
}

template <typename EigenBaseType>
static inline const EigenBaseType
json_to_eigen_bias(std::string &bias_str_no_parentheses, size_t rows) {
  std::vector<double> bias_values;
  bias_values.reserve(rows);

  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  while ((pos = bias_str_no_parentheses.find(delimiter)) != std::string::npos) {
    token = bias_str_no_parentheses.substr(0, pos);
    double value = std::stod(token);
    bias_values.emplace_back(value);
    bias_str_no_parentheses.erase(0, pos + delimiter.length());
  }

  bias_values.emplace_back(std::stod(bias_str_no_parentheses));

  EigenBaseType base(bias_values.size());
  int i = 0;
  std::for_each(bias_values.begin(), bias_values.end(),
                [&base, &i](double t) { base(i++) = t; });
  return base;
}

template <typename EigenBaseType>
static inline const std::vector<double>
json_to_eigen_row(std::string &bias_str_no_parentheses, size_t rows) {
  std::vector<double> bias_values;
  bias_values.reserve(rows);

  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  while ((pos = bias_str_no_parentheses.find(delimiter)) != std::string::npos) {
    token = bias_str_no_parentheses.substr(0, pos);
    double value = std::stod(token);
    bias_values.emplace_back(value);
    bias_str_no_parentheses.erase(0, pos + delimiter.length());
  }

  bias_values.emplace_back(std::stod(bias_str_no_parentheses));

  return bias_values;
}

/**
 * Converts a Eigen Vector to a nested json array.
 * @param weight matrix
 * @return json nested array.
 */
template <typename EigenBaseType = Eigen::VectorXd>
static const inline EigenBaseType json_to_eigen(std::string &str, EigenType t,
                                                size_t rows, size_t cols) {
  auto left_paren_gone = str.substr(1, std::string::npos);
  auto right_paren_gone = left_paren_gone.substr(0, left_paren_gone.size() - 1);

  // We have now removed the nesting parentheses of either side.
  switch (t) {
  case EigenType::VECTOR:
    return json_to_eigen_bias<Eigen::VectorXd>(right_paren_gone, rows);
  case EigenType::MATRIX:
    return json_to_eigen_weight<Eigen::MatrixXd>(right_paren_gone, rows, cols);
  default:
    throw std::runtime_error("Invalid eigen type provided.");
  }
}

#endif // CORTESIAN_DATAREADER_H
