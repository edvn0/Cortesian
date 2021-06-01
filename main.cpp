#include "libs/Eigen/Core"
#include "nn/ArgMaxEval.h"
#include "nn/EarlyStopping.h"
#include "nn/EigenInitializer.h"
#include "nn/Layer.h"
#include "nn/LeakyRelu.h"
#include "nn/LinearFunction.h"
#include "nn/MeanAbsolute.h"
#include "nn/MeanSquared.h"
#include "nn/Network.h"
#include "nn/SigmoidFunction.h"
#include "nn/StochasticGradientDescent.h"
#include "nn/Tanh.h"
#include <chrono>
#include <iostream>
#include <random>

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

  std::random_device rd;
  std::mt19937::result_type seed = rd() ^ (
      (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch()
          ).count() +
      (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch()
          ).count() );

  std::mt19937 gen(seed);
  std::uniform_int_distribution<unsigned> distrib(0, 3);

  BlockTimer t;
  for (int i = 0; i < NUM_DS; i++) {
    int rand = distrib(gen);
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
