#include <Eigen/Core>
#include "nn/ArgMaxEval.h"
#include "nn/EigenInitializer.h"
#include "nn/Layer.h"
#include "nn/LeakyRelu.h"
#include "nn/MeanAbsolute.h"
#include "nn/MeanSquared.h"
#include "nn/Network.h"
#include "nn/StochasticGradientDescent.h"
#include "nn/Tanh.h"
#include <chrono>
#include <iostream>
#include <effolkronium/random.hpp>

using Random = effolkronium::random_static;

#define NUM_DS 100
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
      .layer(Layer(new Tanh(), 2));

  Network network(builder);

  std::vector<Eigen::VectorXd> X;
  X.reserve(NUM_DS);
  std::vector<Eigen::VectorXd> Y;
  Y.reserve(NUM_DS);

  Eigen::Vector2d X_S[] = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
  Eigen::Vector2d Y_S[] = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};

  BlockTimer t;
  for (int i = 0; i < NUM_DS; i++) {
    size_t rand = Random::get(0,3);
    X.emplace_back(X_S[rand]);
    Y.emplace_back(Y_S[rand]);
  }
  t.stop();
  std::cout << "Time taken: " << t.elapsedSeconds() << "\n";

  auto out = network.fit(X, Y, 10, 64);

  std::cout << out;
  std::cout << network.classify(X_S[0]) << "\n" << network.classify(X_S[3]) << "\n";
  return 0;
}
