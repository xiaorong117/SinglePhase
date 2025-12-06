#include "ParametersIni.hpp"
#include "Globals.hpp"        // 引入所有全局参数、结构体和命名空间
#include "Memory.hpp"
#include "MeshInput.hpp"

#include <omp.h>        // For OpenMP functions and macros
#include <cmath>        // For std::pow
#include <iostream>

void ParametersIni::initializePhysicalParameters() {
  // ... 其他初始化方法调用 ...
  calculatePoreVolumeAndFluidProps();
  std::cout << "Physical parameters initialization complete." << std::endl;
};

{
  // 计算孔隙的体积
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;        // 孔隙网络单元
  }

  // 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    // Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 *
    // pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
    Pb[i].compre = 1;
    Pb[i].compre_old = 1;
    Pb[i].visco = kong::viscosity;
    Pb[i].visco_old = kong::viscosity;
  }
}
