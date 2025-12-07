#include "GasPhysicsModel.hpp"
#include <omp.h>
#include <cmath>
#include "Globals.hpp"
#include "MeshInput.hpp"
// For gsl
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_roots.h>

using namespace My_const;
using namespace Fluid_property;
using namespace Porous_media_property_Hybrid;
using namespace Porous_media_property_PNM;
using namespace Physical_property;
using namespace Solver_property;

// 1. 初始化实现
void GasPhysicsModel::initialize() {
  Memory& mem = Memory::getInstance();
  MeshInput& mesh = MeshInput::getInstance();
  pore* Pb = mem.get_Pb();
  int pn = mesh.get_pn();

  // 初始化体积、粘度等 (复用你之前的逻辑)
#pragma omp parallel for
  for (int i = 0; i < pn; i++) {
    Pb[i].volume = 4 * My_const::pi * std::pow(Pb[i].Radiu, 3) / 3;
    Pb[i].visco = Fluid_property::kong::viscosity;        // 初始粘度
    Pb[i].compre = 1.0;                                   // 初始压缩系数
    Pb[i].visco_old = Pb[i].visco;
    Pb[i].compre_old = Pb[i].compre;
  }
}

// 2. 更新实现 (这是你新需求的核心)
void GasPhysicsModel::updateProperties() {
  Memory& mem = Memory::getInstance();
  MeshInput& mesh = MeshInput::getInstance();
  pore* Pb = mem.get_Pb();
  int pn = mesh.get_pn();

  // 示例：假设粘度随压力变化 (这里写你的物理公式)
#pragma omp parallel for
  for (int i = 0; i < pn; i++) {}
}

// 辅助函数可以设为私有
void GasPhysicsModel::Function_DS(double pressure) {
  Ds = (Ds_LIST[6] - Ds_LIST[0]) / (50e6 - 1e6) * (pressure - 1e6) + Ds_LIST[0];
};

double GasPhysicsModel::compre(double pressure) {
  double Tr = Temperature / T_critical;
  double Pr = pressure / P_critical;

  double w_a = 0.45723553;
  double w_b = 0.07779607;
  double w = 0.008;
  double m = 0.37464 + 1.54226 * w - 0.2699 * w * w;

  double A = w_a * pow(1 + m * (1 - sqrt(Tr)), 2) * Pr / pow(Tr, 2);
  double B = w_b * Pr / Tr;

  double solutions[3]{-1e4, -1e4, -1e4};
  gsl_poly_solve_cubic(B - 1, A - 3 * B * B - 2 * B, -A * B + B * B + B * B * B, &solutions[0], &solutions[1], &solutions[2]);

  if ((int)solutions[2] != -1e4) {
    std::sort(solutions, solutions + 3, std::greater<double>());
    return solutions[0];
  } else {
    return solutions[0];
  }
};

double GasPhysicsModel::visco(double p, double z, double T) {
  p = 0.00014504 * p;                                                             // pa -> psi
  T = 1.8 * T;                                                                    // k -> Rankin
  double density_of_gas = 28.967 * 0.5537 * p / (z * 10.732 * T) / 62.428;        // g/cm3
  double Mg = 28.967 * 0.5537;
  double X = 3.448 + 986.4 / (T) + 0.001 * Mg;        // T in R, M in g/mol
  double Y = 2.447 - 0.2224 * X;
  double K = (9.379 + 0.02 * Mg) * pow(T, 1.5) / (209.2 + 19.26 * Mg + T);
  return 1e-7 * K * exp(X * pow(density_of_gas, Y));        // cp -> Pa s
};

double GasPhysicsModel::Function_Slip(double knusen) {
  double alpha_om = 1.358 * 2 / pi * atan(4 * pow(knusen, 0.4));
  double beta_om = 4;
  double Slip_om = (1 + alpha_om * knusen) * (1 + beta_om * knusen / (1 + knusen));
  return Slip_om;
}

double GasPhysicsModel::Function_Slip_clay(double knusen) {
  double alpha_c = 1.5272 * 2 / pi * atan(2.5 * pow(knusen, 0.5));
  double beta_c = 6;
  double Slip_c = (1 + alpha_c * knusen) * (1 + beta_c * knusen / (1 + knusen));
  return Slip_c;
}