cmake_minimum_required(VERSION 3.19)
project(cortesian)

set(Eigen3_DIR="/Volumes/Toshiba/GitBuilds/eigen")
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(effolkronium_random REQUIRED)
find_package(Catch2 REQUIRED)

add_subdirectory(libs/csv-parser)

add_executable(
  cortesian
  main.cpp
        include/activations/Activation.h
        src/activations/LeakyRelu.cpp
        include/activations/LeakyRelu.h
        src/layers/Dense.cpp
        include/layers/Dense.h
        include/optimizers/Optimizer.h
        src/optimizer/StochasticGradientDescent.cpp
        include/optimizers/StochasticGradientDescent.h
  src/Network.cpp
  include/Network.h
        include/loss_evals/LossFunction.h
        include/loss_evals/EvaluationFunction.h
        include/initializers/ParameterInitializer.h
  include/NetworkBuilder.h
        src/util/BackPropStatistics.cpp
        include/utils/BackPropStatistics.h
        src/util/BlockTimer.cpp
        include/utils/BlockTimer.h
        src/initializers/EigenInitializer.cpp
        include/initializers/EigenInitializer.h
        src/loss_eval/ArgMax.cpp
        include/utils/ArgMax.h
        src/loss_eval/MeanSquared.cpp
        include/loss_evals/MeanSquared.h
        include/utils/MathUtils.h
        src/util/DataSplit.cpp
        include/utils/DataSplit.h
        src/activations/LinearFunction.cpp
        include/activations/LinearFunction.h
        src/loss_eval/MeanAbsolute.cpp
        include/loss_evals/MeanAbsolute.h
        src/activations/SigmoidFunction.cpp
        include/activations/SigmoidFunction.h
        src/activations/Tanh.cpp
        include/activations/Tanh.h
        src/activations/Softmax.cpp
        include/activations/Softmax.h
        include/utils/common.h
        include/utils/DataReader.h
        src/loss_eval/CategoricalCrossEntropy.cpp
        include/loss_evals/CategoricalCrossEntropy.h
        src/optimizer/Adam.cpp
        include/optimizers/Adam.h include/layers/Layer.h)
target_link_libraries(cortesian effolkronium_random Eigen3::Eigen csv)

add_executable(cortesian_tests tests/test.cpp)
target_link_libraries(cortesian_tests PRIVATE Catch2::Catch2WithMain)
