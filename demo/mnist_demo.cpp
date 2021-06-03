#include <eigen3/Eigen/Core>
#include <chrono>
#include <iostream>

#include "../include/Network.h"
#include "../include/activations/LeakyRelu.h"
#include "../include/activations/Softmax.h"
#include "../include/initializers/EigenInitializer.h"
#include "../include/loss_evals/ArgMax.h"
#include "../include/loss_evals/CategoricalCrossEntropy.h"
#include "../include/loss_evals/MeanAbsolute.h"
#include "../include/loss_evals/MeanSquared.h"
#include "../include/optimizers/Adam.h"
#include "../include/utils/DataReader.h"

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
          {new ArgMax(), new MeanAbsolute(), new MeanSquared()})
      .initializer(new EigenInitializer())
      .optimizer(new Adam(1))
      .layer(new Dense(new LeakyRelu(), 784, 0.1))
      .layer(new Dense(new LeakyRelu(), 300, 0.5))
      .layer(new Dense(new LeakyRelu(), 300, 0.5))
      .layer(new Dense(new LeakyRelu(), 300, 0.5))
      .layer(new Dense(new LeakyRelu(), 300, 0.5))
      .layer(new Dense(new Softmax(), 10, 0.5));

  Network network(builder);

  std::cout << network;

  auto out = network.fit_tensor(X_tensor, Y_tensor, 100, 64, X_validate_tensor,
                                Y_validate_tensor);

  std::cout << out;
  return 0;
}
