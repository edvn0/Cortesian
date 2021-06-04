# Cortesian

Cortesian is a learning project stemming from an implementation I built in Java 2020-2021.

I intend to provide transparency to how neural networks work and why they perform well on problems, and how you could implement one yourself!

Starting in Java, I now intend to move almost exclusively to C++, which means that I need to understand the language and its intricacies.

The code is far from performant, perfect, good, or even decent, however, I do not concern myself with those criteria in this project.

My goal for this project is to be able to create a convolutional neural network which can get ~80% on Cifar-100.

If you wish to help me or comment the code, please do, and submit changes as pull requests.

## API

A `Network` is created by a `NetworkBuilder`, like so:

```
NetworkBuilder builder;
// Configuration here.
builder.loss(new MeanSquared())
    .initializer(new EigenInitializer())
    .optimizer(new StochasticGradientDescent())
    .evaluation({ new MeanAbsolute() });

// Instead of providing an initializer list to evaluation
// you can provide a single evaluation function like so:
builder.evaluation(new MeanAbsolute());

// We also need to define layers.
```

Now, we need to add `Layer`. Layers need to represent your data. If you add 4 layers, that means that there are 3 weight/bias pairs connecting them, effectively meaning that you have 1
input layer, 2 hidden layers and 1 output layer.

```
// Currently only accepts vectors as inputs, 
// unlike Keras taking tensors.
size_t input_size; 

// How many neurons are there in the hidden layers? 
size_t hidden_neurons_first;
size_t hidden_neurons_second;

// Currently only accepts vectors as outputs, 
// unlike Keras taking tensors.
size_t output_size; 
// Relu capping, setting the values of the activated  
// values to this value if it's less than zero.
double relu_cap = 0.01;

// Statistical property, weighting single samples
// against the loss of the entire network.
double l2_regularization = 0.2; 

builder.layer(new Layer(input_nodes, new LeakyRelu(relu_cap), relu_cap))
    .layer(new Layer(hidden_neurons_first, new LeakyRelu(relu_cap), relu_cap))
    .layer(new Layer(hidden_neurons_second, new LeakyRelu(relu_cap), relu_cap))
    .layer(new Layer(output_size, new LeakyRelu(relu_cap), relu_cap));
```

Finally, to construct the network:

```
Network network(builder); // constructs the network. 
// builds weights/biases etc.
```

To use the network you need data matching your configuration. This can be loaded via:

```
csv_to_tensor(const std::string &file_name, 
              size_t rows, 
              size_t X_cols, 
              size_t Y_cols,
              const std::function<void(Eigen::MatrixXd &, csv::CSVField &,size_t, size_t)> &X_mapper,
              const std::function<void(Eigen::MatrixXd &, csv::CSVField &,size_t, size_t)> &Y_mapper,
              bool has_header = true, 
              char delimiter = ',')
```

The mappers are functors applied to each row, receiving the output tensor, the current field of your csv file, the row index currently to be changed and the column index currently to be changed.
An example of an invocation of this function with mnist data:

```
// Since MNIST data infamously has 785 values per row, 
// the class being the 0th value, we need to check for this in the functors.

// This is obviously a lousy way to handle tensor loading from csv.
// This now occupies about 700mb of space...
auto [X_tensor, Y_tensor] = csv_to_tensor(
  "resources/mnist_train.csv",
  60000, 784, 10,
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
```
