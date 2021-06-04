
//
// Created by Edwin Carlsson on 2021-06-02.
//

#include <catch2/catch_all.hpp>

#include "../include/utils/DataReader.h"
#include "../src/activations/Softmax.cpp"
#include "../src/loss_eval/CategoricalCrossEntropy.cpp"

#define CORTESIAN_TEST
#define CATCH_CONFIG_DISABLE_BENCHMARKING

static constexpr double certainty = 1e-6;

TEST_CASE("Softmax") {
  SECTION("Should correctly handle case 0.6,0.2,0.2") {
    Activation *f = new Softmax();
    Eigen::VectorXd vec(10);
    vec << 0.0, 0.00, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.6;
    Eigen::VectorXd expected(10);
    expected << 0.08877112, 0.08877112, 0.08877112, 0.08877112, 0.1084253,
        0.1084253, 0.08877112, 0.08877112, 0.08877112, 0.16175154;
    auto calc = f->function(vec);
    REQUIRE((calc - expected).norm() < certainty);
  }

  SECTION("Should correctly handle case HUGE certainty") {
    Activation *f = new Softmax();
    Eigen::VectorXd vec(10);
    vec << 0.0, 0.00, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 100;
    Eigen::VectorXd expected(10);
    expected << 0.0, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1;
    auto calc = f->function(vec);
    REQUIRE((calc - expected).norm() < certainty);
  }
}

TEST_CASE("CategoricalCrossEntropy") {
  SECTION("Should return ~0.708 when tested on particular values") {
    LossFunction *f = new CategoricalCrossEntropy();
    std::vector<Eigen::VectorXd> preds;
    Eigen::Vector<double, 4> v1, v2;
    v1 << 0.25, 0.25, 0.25, 0.25;
    v2 << 0.01, 0.01, 0.01, 0.97;
    preds.emplace_back(v1);
    preds.emplace_back(v2);

    Eigen::Matrix<double, 2, 4> actual;
    actual.row(0) << 1.0, 0.0, 0.0, 0.0;
    actual.row(1) << 0.0, 0.0, 0.0, 1.0;

    double calculated = f->apply_loss(preds, actual);
    std::cout << calculated;
    REQUIRE(abs(calculated - 0.7083767843022996) < certainty);
    delete f;
  }
}

TEST_CASE("Eigen Serialization/Deserialization") {
  SECTION("S/D Vector") {
    Eigen::VectorXd vec(3);
    vec << 1.000000012, 2.00000151, 3.0000000051512;
    std::string ser = eigen_to_json(vec);
    Eigen::VectorXd deser =
        json_to_eigen<Eigen::VectorXd>(ser, EigenType::VECTOR, 3, 1);

    REQUIRE((deser - vec).norm() < certainty);
  }

  SECTION("S/D Random 100 vector") {
    Eigen::VectorXd vec = Eigen::VectorXd::Random(100);
    std::string ser = eigen_to_json(vec);
    Eigen::VectorXd deser =
        json_to_eigen<Eigen::VectorXd>(ser, EigenType::VECTOR, 100, 1);

    REQUIRE((deser - vec).norm() < certainty);
  }

  SECTION("S/D Matrix (2,2)") {
    Eigen::MatrixXd mat(2, 2);
    mat << 1.000001231312, 2.021412321, 3.12321321, 4.12521;
    std::string ser = eigen_to_json(mat);
    Eigen::MatrixXd deser =
        json_to_eigen<Eigen::MatrixXd>(ser, EigenType::MATRIX, 2, 2);

    REQUIRE((deser - mat).norm() < certainty);
  }

  SECTION("S/D Matrix (1000,341)") {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(1000, 341);
    std::string ser = eigen_to_json(mat);
    Eigen::MatrixXd deser =
        json_to_eigen<Eigen::MatrixXd>(ser, EigenType::MATRIX, 1000, 341);

    REQUIRE((deser - mat).norm() < certainty);
  }

  SECTION("S/D Matrix (341,1000)") {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(341, 1000);
    std::string ser = eigen_to_json(mat);
    Eigen::MatrixXd deser =
        json_to_eigen<Eigen::MatrixXd>(ser, EigenType::MATRIX, 341, 1000);

    REQUIRE((deser - mat).norm() < certainty);
  }
}

TEST_CASE("BENCHMARKS Json Deser/Ser", "[benchjson]") {
  // Average 159ms
  BENCHMARK("Stress test S/D weights (256,256)") {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(256, 256);
    std::string ser = eigen_to_json(mat);
    return json_to_eigen<Eigen::MatrixXd>(ser, EigenType::MATRIX, 256, 256);
  };
}