#include "libs/Eigen/Core"
#include "nn/ArgMaxEval.h"
#include "nn/EigenInitializer.h"
#include "nn/Layer.h"
#include "nn/LeakyRelu.h"
#include "nn/LinearFunction.h"
#include "nn/MeanAbsolute.h"
#include "nn/MeanSquared.h"
#include "nn/Network.h"
#include "nn/SigmoidFunction.h"
#include "nn/StochasticGradientDescent.h"
#include <iostream>

#define NUM_DS 10000
int main() {

  NetworkBuilder builder;
  builder.clipping(0.5)
      .loss_function(new MeanSquared())
      .evaluation_function(
          {new ArgMaxEval(), new MeanAbsolute(), new MeanSquared()})
      .initializer(new EigenInitializer())
      .optimizer(new StochasticGradientDescent(0.1))
      .layer(Layer(new LeakyRelu(), 2, 0.9))
      .layer(Layer(new LeakyRelu(), 30, 0.9))
      .layer(Layer(new LeakyRelu(), 30, 0.9))
      .layer(Layer(new SigmoidFunction(), 2));

  Network network(builder);

  std::vector<Eigen::VectorXd> X;
  X.reserve(NUM_DS);
  std::vector<Eigen::VectorXd> Y;
  Y.reserve(NUM_DS);

  Eigen::Vector2d X_S[] = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Eigen::Vector2d Y_S[] = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};

  for (int i = 0; i < NUM_DS; i++) {
    X.emplace_back(X_S[i % 4]);
    Y.emplace_back(Y_S[i % 4]);
  }

  auto out = network.fit(X, Y, 40, 64);

  std::cout << out;

  std::cout << network << "\n";

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
