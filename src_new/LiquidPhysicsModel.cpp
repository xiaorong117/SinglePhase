#include "LiquidPhysicsModel.hpp"
#include <omp.h>
#include <cmath>
#include "Globals.hpp"
#include "MeshInput.hpp"

using namespace My_const;
using namespace Fluid_property;
using namespace Porous_media_property_Hybrid;
using namespace Porous_media_property_PNM;
using namespace Physical_property;
using namespace Solver_property;

// 1. 初始化实现
void LiquidPhysicsModel::initialize() {
  Memory& mem = Memory::getInstance();
  MeshInput& mesh = MeshInput::getInstance();
  pore* Pb = mem.get_Pb();
  int pn = mesh.get_pn();

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
void LiquidPhysicsModel::updateProperties() {
  Memory& mem = Memory::getInstance();
  MeshInput& mesh = MeshInput::getInstance();
  pore* Pb = mem.get_Pb();
  int pn = mesh.get_pn();
}

// 辅助函数可以设为私有

double LiquidPhysicsModel::compre(double pressure) {
  return 1;
};

double LiquidPhysicsModel::visco(double p, double z, double T) {
  return 2e-5;
};