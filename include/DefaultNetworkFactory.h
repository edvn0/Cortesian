//
// Created by Edwin Carlsson on 2021-06-03.
//

#ifndef CORTESIAN_DEFAULTNETWORKFACTORY_H
#define CORTESIAN_DEFAULTNETWORKFACTORY_H

#include "Network.h"
#include "activations/LeakyRelu.h"
#include "activations/Sigmoid.h"
#include "activations/Softmax.h"
#include "initializers/EigenInitializer.h"
#include "layers/Dense.h"
#include "loss_evals/ArgMax.h"
#include "loss_evals/CategoricalCrossEntropy.h"
#include "loss_evals/MeanAbsolute.h"
#include "loss_evals/MeanSquared.h"
#include "optimizers/Adam.h"

static Network multi_layer_perceptron(size_t input_neurons, size_t layers,
                                      size_t hidden_neurons,
                                      size_t output_neurons,
                                      bool is_classifier = true) {
  NetworkBuilder builder;
  builder.clipping(0.5)
      .initializer(new EigenInitializer())
      .optimizer(new Adam(0.01))
      .layer(new Dense(new LeakyRelu(), (int)input_neurons, 0.1));

  for (size_t t = 0; t < layers; t++) {
    builder.layer(new Dense(new LeakyRelu(), (int)hidden_neurons, 0.1));
  }

  if (is_classifier) {
    builder.layer(new Dense(new Softmax(), (int)output_neurons, 0.1));
    builder.loss_function(new CategoricalCrossEntropy());
    builder.evaluation_function(
        {new ArgMax(), new MeanAbsolute(), new MeanSquared()});
  } else {
    builder.layer(new Dense(new Sigmoid(), (int)output_neurons, 0.1));
    builder.loss_function(new CategoricalCrossEntropy());
    builder.evaluation_function({new MeanAbsolute(), new MeanSquared()});
  }

  return Network(builder);
}

static Network perceptron(size_t input_neurons, size_t hidden_neurons,
                          size_t output_neurons, bool is_classifier = true) {
  NetworkBuilder builder;
  builder.clipping(0.5)
      .initializer(new EigenInitializer())
      .optimizer(new Adam(0.01))
      .layer(new Dense(new LeakyRelu(), (int)input_neurons, 0.1))
      .layer(new Dense(new LeakyRelu(), (int)hidden_neurons, 0.1));

  if (is_classifier) {
    builder.layer(new Dense(new Softmax(), (int)output_neurons, 0.1));
    builder.loss_function(new CategoricalCrossEntropy());
    builder.evaluation_function(
        {new ArgMax(), new MeanAbsolute(), new MeanSquared()});
  } else {
    builder.layer(new Dense(new Sigmoid(), (int)output_neurons, 0.1));
    builder.loss_function(new CategoricalCrossEntropy());
    builder.evaluation_function({new MeanAbsolute(), new MeanSquared()});
  }

  return Network(builder);
}

#endif // CORTESIAN_DEFAULTNETWORKFACTORY_H
