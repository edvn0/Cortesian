cmake_minimum_required(VERSION 3.19)
project(cortesian)

set(CMAKE_CXX_STANDARD 17)
file(GLOB libs/Eigen CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(cortesian main.cpp nn/Activation.h nn/LeakyRelu.cpp nn/LeakyRelu.h nn/Layer.cpp nn/Layer.h nn/Optimizer.h nn/StochasticGradientDescent.cpp nn/StochasticGradientDescent.h nn/Network.cpp nn/Network.h nn/LossFunction.h nn/EvaluationFunction.h nn/ParameterInitializer.h nn/NetworkBuilder.h nn/BackPropStatistics.cpp nn/BackPropStatistics.h nn/BlockTimer.cpp nn/BlockTimer.h nn/EigenInitializer.cpp nn/EigenInitializer.h nn/ArgMaxEval.cpp nn/ArgMaxEval.h nn/MeanSquared.cpp nn/MeanSquared.h nn/MathUtils.h nn/DataSplit.cpp nn/DataSplit.h nn/LinearFunction.cpp nn/LinearFunction.h nn/MeanAbsolute.cpp nn/MeanAbsolute.h nn/SigmoidFunction.cpp nn/SigmoidFunction.h)