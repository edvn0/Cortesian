#include <Eigen/Core>
#include <chrono>
#include <effolkronium/random.hpp>
#include <iostream>

#include "libs/csv-parser/single_include/csv.hpp"
#include "nn/ArgMaxEval.h"
#include "nn/DataReader.h"
#include "nn/EigenInitializer.h"
#include "nn/Layer.h"
#include "nn/LeakyRelu.h"
#include "nn/MeanAbsolute.h"
#include "nn/MeanSquared.h"
#include "nn/Network.h"
#include "nn/Softmax.h"
#include "nn/StochasticGradientDescent.h"

int main() {
  auto [X_tensor, Y_tensor] = csv_to_tensor(
      "/Users/edwincarlsson/Documents/Code.nosync/CPP/DeepLearning/cortesian/"
      "resources/mnist_train.csv",
      60000, 784, 10,
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

  auto [X_validate_tensor, Y_validate_tensor] = csv_to_tensor(
      "/Users/edwincarlsson/Documents/Code.nosync/CPP/DeepLearning/cortesian/"
      "resources/mnist_test.csv",
      10000, 784, 10,
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
      .loss_function(new MeanSquared())
      .evaluation_function(
          {new ArgMaxEval(), new MeanAbsolute(), new MeanSquared()})
      .initializer(new EigenInitializer())
      .optimizer(new StochasticGradientDescent(0.001))
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
