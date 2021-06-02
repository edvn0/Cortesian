//
// Created by Edwin Carlsson on 2021-06-02.
//

#define CONFIG_CATCH_MAIN
#include <catch2/catch_all.hpp>

#include "../src/CategoricalCrossEntropy.cpp"

TEST_CASE("CGE should return 0") {
  LossFunction* f = new CategoricalCrossEntropy();
}