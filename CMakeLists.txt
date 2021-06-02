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
  include/Activation.h
  src/LeakyRelu.cpp
  include/LeakyRelu.h
  src/Layer.cpp
  include/Layer.h
  include/Optimizer.h
  src/StochasticGradientDescent.cpp
  include/StochasticGradientDescent.h
  src/Network.cpp
  include/Network.h
  include/LossFunction.h
  include/EvaluationFunction.h
  include/ParameterInitializer.h
  include/NetworkBuilder.h
  src/BackPropStatistics.cpp
  include/BackPropStatistics.h
  src/BlockTimer.cpp
  include/BlockTimer.h
  src/EigenInitializer.cpp
  include/EigenInitializer.h
  src/ArgMaxEval.cpp
  include/ArgMaxEval.h
  src/MeanSquared.cpp
  include/MeanSquared.h
  include/MathUtils.h
  src/DataSplit.cpp
  include/DataSplit.h
  src/LinearFunction.cpp
  include/LinearFunction.h
  src/MeanAbsolute.cpp
  include/MeanAbsolute.h
  src/SigmoidFunction.cpp
  include/SigmoidFunction.h
  src/Tanh.cpp
  include/Tanh.h
  src/Softmax.cpp
  include/Softmax.h
  include/common.h
  include/DataReader.h
  src/CategoricalCrossEntropy.cpp
  include/CategoricalCrossEntropy.h
  src/Adam.cpp
  include/Adam.h)
target_link_libraries(cortesian effolkronium_random Eigen3::Eigen csv)

add_executable(cortesian_tests tests/test.cpp)
target_link_libraries(cortesian_tests PRIVATE Catch2::Catch2)
