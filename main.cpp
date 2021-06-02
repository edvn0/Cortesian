#include <Eigen/Core>
#include <chrono>
#include <effolkronium/random.hpp>
#include <iostream>

#include "include/Adam.h"
#include "include/ArgMaxEval.h"
#include "include/CategoricalCrossEntropy.h"
#include "include/DataReader.h"
#include "include/EigenInitializer.h"
#include "include/Layer.h"
#include "include/LeakyRelu.h"
#include "include/MeanAbsolute.h"
#include "include/MeanSquared.h"
#include "include/Network.h"
#include "include/Softmax.h"
#include "include/StochasticGradientDescent.h"
#include "libs/csv-parser/single_include/csv.hpp"

int main() {
  auto [X_tensor, Y_tensor] = csv_to_tensor(
      "/Users/edwincarlsson/Documents/Code.nosync/CPP/DeepLearning/cortesian/"
      "resources/mnist_case_x.csv",
      768, 784, 10,
      [](Eigen::MatrixXd &matrix, csv::CSVField &field, size_t row,
         size_t col) {
        if (col != 0) {
          matrix((long)row, (long)col - 1) = field.get<double>() / 255.0;
        }
      },
      [](Eigen::MatrixXd &matrix, csv::CSVField &field, size_t row,
         size_t col) {
        if (col == 0) {
          matrix((long)row, field.get<int>()) = 1.0;
        }
      });

  auto [X_validate_tensor, Y_validate_tensor] = csv_to_tensor(
      "/Users/edwincarlsson/Documents/Code.nosync/CPP/DeepLearning/cortesian/"
      "resources/mnist_case_y.csv",
      298, 784, 10,
      [](Eigen::MatrixXd &matrix, csv::CSVField &field, size_t row,
         size_t col) {
        if (col != 0) {
          matrix((long)row, (long)col - 1) = field.get<double>();
        }
      },
      [](Eigen::MatrixXd &matrix, csv::CSVField &field, size_t row,
         size_t col) {
        if (col == 0) {
          matrix((long)row, field.get<int>()) = 1.0;
        }
      });

  NetworkBuilder builder;
  builder.clipping(0.5)
      .loss_function(new CategoricalCrossEntropy())
      .evaluation_function(
          {new ArgMaxEval(), new MeanAbsolute(), new MeanSquared()})
      .initializer(new EigenInitializer())
      .optimizer(new Adam(0.0001))
      .layer(Layer(new LeakyRelu(), 784, 0.1))
      .layer(Layer(new LeakyRelu(), 30, 0.1))
      .layer(Layer(new LeakyRelu(), 30, 0.1))
      .layer(Layer(new Softmax(), 10));

  Network network(builder);

  auto out = network.fit_tensor(X_tensor, Y_tensor, 10, 64, X_validate_tensor,
                                Y_validate_tensor);

  std::cout << out;
  return 0;
}
