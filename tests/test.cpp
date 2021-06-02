//
// Created by Edwin Carlsson on 2021-06-02.
//

#include <catch2/catch_all.hpp>

#include "../src/loss_eval/CategoricalCrossEntropy.cpp"

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