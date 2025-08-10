#include <gtest/gtest.h>
#include "Solver_openmp_ad.cpp"

class PNMsolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data or objects here
    }

    void TearDown() override {
        // Clean up test data or objects here
    }
};

TEST_F(PNMsolverTest, TestConductivitySurTest) {
    PNMsolver solver;
    reverse_mode<double> Pi = 1.0;
    reverse_mode<double> Pjs[1] = {0.5};
    reverse_mode<double> Wi = 0.1;
    reverse_mode<double> Wjs[1] = {0.05};
    int pore_id = 0;
    int throat_id = 0;

    reverse_mode<double> result = solver.conductivity_sur_test(Pi, Pjs, Wi, Wjs, pore_id, throat_id);
    EXPECT_GT(result.x(), 0.0);
}

TEST_F(PNMsolverTest, TestConductivityBulkTest) {
    PNMsolver solver;
    reverse_mode<double> Pi = 1.0;
    reverse_mode<double> Pjs[1] = {0.5};
    reverse_mode<double> Wi = 0.1;
    reverse_mode<double> Wjs[1] = {0.05};
    int pore_id = 0;
    int throat_id = 0;

    reverse_mode<double> result = solver.conductivity_bulk_test(Pi, Pjs, Wi, Wjs, pore_id, throat_id);
    EXPECT_GT(result.x(), 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}