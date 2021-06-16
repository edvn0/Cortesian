#include <chrono>
#include <eigen3/Eigen/Core>
#include <iostream>

#include "../include/Network.h"
#include "../include/activations/LeakyRelu.h"
#include "../include/activations/Softmax.h"
#include "../include/activations/Tanh.h"
#include "../include/initializers/EigenInitializer.h"
#include "../include/loss_evals/ArgMax.h"
#include "../include/loss_evals/CategoricalCrossEntropy.h"
#include "../include/loss_evals/MeanAbsolute.h"
#include "../include/loss_evals/MeanSquared.h"
#include "../include/loss_evals/LogLoss.h"
#include "../include/optimizers/Adam.h"
#include "../include/optimizers/StochasticGradientDescent.h"
#include "../include/utils/DataReader.h"

int main()
{
    auto [X_tensor, Y_tensor] = csv_to_mnist(
        "/Users/edwincarlsson/Downloads/"
        "mnist_train.csv",
        784, 10, 6000);

    auto [X_validate_tensor, Y_validate_tensor] = csv_to_mnist(
        "/Users/edwincarlsson/Downloads/"
        "mnist_test.csv",
        784, 10, 1000);

    NetworkBuilder builder;
    builder
        .loss_function(new LogLoss())
        .evaluation_function(
            {new ArgMax(), new MeanAbsolute(), new MeanSquared()})
        .initializer(new EigenInitializer())
        .optimizer(new Adam(0.0001))
        .layer(new Dense(new LeakyRelu(0.01), 784, 0.4))
        .layer(new Dense(new LeakyRelu(0.01), 50, 0.1))
        .layer(new Dense(new LeakyRelu(0.01), 50, 0.1))
        .layer(new Dense(new Softmax(), 10, 0.4));

    Network network(builder);

    std::cout << network;

    network.save("../resources/model.json");

    auto out = network.fit_tensor(X_tensor, Y_tensor, 100, 12, X_validate_tensor,
                                  Y_validate_tensor);

    std::cout << out;
    return 0;
}
