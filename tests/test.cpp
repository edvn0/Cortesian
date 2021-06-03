
//
// Created by Edwin Carlsson on 2021-06-02.
//

#include <catch2/catch_all.hpp>

#include "../src/activations/Softmax.cpp"
#include "../src/loss_eval/CategoricalCrossEntropy.cpp"

TEST_CASE("Softmax") {
  SECTION("Should correctly handle case 0.6,0.2,0.2") {
    Activation *f = new Softmax();
    Eigen::VectorXd vec(10);
    vec << 0.0, 0.00, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.6;
    Eigen::VectorXd expected(10);
    expected << 0.08877112, 0.08877112, 0.08877112, 0.08877112, 0.1084253,
        0.1084253, 0.08877112, 0.08877112, 0.08877112, 0.16175154;
    auto calc = f->function(vec);
    REQUIRE((calc-expected).norm() < 0.00001);
  }

  SECTION("Should correctly handle case HUGE certainty") {
    Activation *f = new Softmax();
    Eigen::VectorXd vec(10);
    vec << 0.0, 0.00, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 100;
    Eigen::VectorXd expected(10);
    expected << 0.0, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1;
    auto calc = f->function(vec);
    std::cout << calc;
    REQUIRE((calc-expected).norm() < 0.00001);
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
    REQUIRE(calculated == 0.7083767843022996);
    delete f;
  }
}