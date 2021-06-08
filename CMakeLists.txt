cmake_minimum_required(VERSION 3.19)
project(cortesian)

set(Eigen3_DIR="/Volumes/Toshiba/GitBuilds/eigen")
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(effolkronium_random REQUIRED)
find_package(Catch2 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_subdirectory(libs/csv-parser)

add_executable(
        cortesian
        demo/mnist_demo.cpp
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
        include/loss_evals/ArgMax.h
        src/loss_eval/MeanSquared.cpp
        include/loss_evals/MeanSquared.h
        include/utils/MathUtils.h
        src/util/DataSplit.cpp
        include/utils/DataSplit.h
        src/activations/Linear.cpp
        include/activations/Linear.h
        src/loss_eval/MeanAbsolute.cpp
        include/loss_evals/MeanAbsolute.h
        src/activations/Sigmoid.cpp
        include/activations/Sigmoid.h
        src/activations/Tanh.cpp
        include/activations/Tanh.h
        src/activations/Softmax.cpp
        include/activations/Softmax.h
        include/utils/common.h
        include/utils/DataReader.h
        src/loss_eval/CategoricalCrossEntropy.cpp
        include/loss_evals/CategoricalCrossEntropy.h
        src/optimizer/Adam.cpp
        include/optimizers/Adam.h
        include/layers/Layer.h
        include/layers/Conv2d.h
        src/layers/Conv2d.cpp
        src/layers/Conv1d.cpp
        include/layers/Conv1d.h
        include/DefaultNetworkFactory.h
        include/utils/MetaBase.h demo/tiny_demo.cpp src/loss_eval/LogLoss.cpp src/loss_eval/LogLoss.h)
target_link_libraries(cortesian effolkronium_random Eigen3::Eigen csv
        OpenMP::OpenMP_CXX)

add_executable(cortesian_tests tests/test.cpp)
target_link_libraries(cortesian_tests PRIVATE Catch2::Catch2WithMain
        OpenMP::OpenMP_CXX)
