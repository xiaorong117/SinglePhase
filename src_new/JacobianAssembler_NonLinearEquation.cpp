#include "JacobianAssembler_NonLinearEquation.hpp"
#include <iostream>
#include <stdexcept>
// 必须包含所有具体的 MatrixConfig 头文件，以便进行 dynamic_cast
#include "LiquidMatrixConfig.hpp"
// #include "GasMatrixConfig.hpp" // 如有其他流体，也应包含
#include "Memory.hpp"
#include "MeshInput.hpp"
#include "Physical_property_helper.hpp"        // 假设边界函数需要这些物理助手
#include "SelfDefinedFunctions.hpp"

template <typename T>
using reverse_mode = B<T>;

using namespace Solver_property;
using namespace Physical_property;
using namespace std;

// ----------------------------------------------------------------------
// 辅助函数：推断 N_VAR (基于配置策略)
// ----------------------------------------------------------------------
int JacobianAssembler_NonLinearEquation::deduce_n_var_per_node(const SystemMatrixConfig& config) const {
  if (dynamic_cast<const LiquidMatrixConfig_PressureOnly*>(&config) || dynamic_cast<const LiquidMatrixConfig_PressureOnly_Neumann*>(&config)) {
    return 1;        // P
  } else if (dynamic_cast<const LiquidMatrixConfig_2Transport_Dirichlet*>(&config) || dynamic_cast<const LiquidMatrixConfig_2Transport_Neumann*>(&config)) {
    return 3;        // P, C1, C2 (双组分运移)
  }
  // ... 未来扩展点 ...
  cerr << "ERROR: Unknown MatrixConfig type! Cannot deduce N_VAR." << endl;
  return 0;
}

// ----------------------------------------------------------------------
// 构造函数
// ----------------------------------------------------------------------
JacobianAssembler_NonLinearEquation::JacobianAssembler_NonLinearEquation(const SystemMatrixConfig& config)
    : config_(config), op_(MeshInput::getInstance().get_op()), mp_(MeshInput::getInstance().get_mp()), n_var_per_node_(deduce_n_var_per_node(config)) {
  if (n_var_per_node_ == 0) {
    throw runtime_error("Pnm Jacobian Assembler failed to initialize: Invalid or unknown configuration type.");
  }
}

// ----------------------------------------------------------------------
// 专门的边界组装函数接口 (需要在 SelfDefinedFunctions.hpp 或专门的头文件中声明)
// ----------------------------------------------------------------------

/**
 * @brief 组装 P + 2Transport (N_VAR=3) 的 Neumann 边界方程。
 * * ⚠️ 这是一个需要在外部实现、封装您原文件边界逻辑的函数。
 * 它负责将额外的边界方程填充到矩阵行 (bc_start_row 及以后) 中。
 */
void assembleNeumannBC_2Transport(double dt, Memory& mem, MeshInput& mesh_data, pore* Pb, throatmerge* Tb, int bc_start_row, int op_mp_total, int N_ROWS) {
  // 提取原文件 JacobianAssembler_LiquidMatrixConfig_2Transport_Neumann.cpp
  // 中关于 inlet, m_inlet 流量平衡的组装逻辑到这里。

  // 边界组装逻辑高度依赖具体的变量、边界数量和内存布局。
  // ...
  // ... (复杂的边界组装逻辑：调用 calculateP_BC_Residuals, calculateC_BC_Residuals 等)
  // ...
}

/**
 * @brief 组装 P Only (N_VAR=1) 的 Neumann 边界方程。
 */
void assembleNeumannBC_POnly(double dt, Memory& mem, MeshInput& mesh_data, pore* Pb, throatmerge* Tb, int bc_start_row, int op_mp_total, int N_ROWS) {
  // 提取原文件 JacobianAssembler_LiquidMatrixConfig_PressureOnly_Neumann.cpp
  // 中关于边界的组装逻辑到这里。
  // ...
}

// ----------------------------------------------------------------------
// 核心组装函数：assemble(dt)
// ----------------------------------------------------------------------
void PnmJacobianAssembler::assemble(double dt) {
  // 1. 获取尺寸和数据指针
  const int N_VAR = n_var_per_node_;
  Memory& mem = Memory::getInstance();
  MeshInput& mesh_data = MeshInput::getInstance();

  const int inlet = mesh_data.get_inlet();
  const int outlet = mesh_data.get_outlet();
  const int m_inlet = mesh_data.get_m_inlet();
  const int m_outlet = mesh_data.get_m_outlet();

  const int op_mp_total = op_ + mp_;

  const int N_ROWS = config_.get_n_variables(mesh_data);
  const int total_nnz = config_.calculate_nnz(mesh_data);

  // ... (数据指针获取，如 Pb, COO_A, B, Tb 等) ...
  pore* Pb = mem.get_Pb();
  Acoo* COO_A = mem.get_COO_A();
  double* B = mem.get_B();
  throatmerge* Tb = mem.get_Tb();

  // 2. 初始化 B 向量和 COO_A 矩阵
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < total_nnz; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
  // mem.reset_COO_counter();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N_ROWS; i++) {
    B[i] = 0;
  }

  // ----------------------------------------------------
  // I. 核心组装循环：处理所有 OP 和 MP 孔隙的系统方程 (通用且固定)
  // ----------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].C1;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      /*这段被注释掉的程序是为了计算渗透率*/
      // if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
      //   B[i - inlet] = 0;
      // } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
      //   B[i - inlet] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      // }
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);
    B[i - inlet] = -F.val();

    COO_A[i - inlet].row = i - inlet;
    COO_A[i - inlet].col = i - inlet;
    COO_A[i - inlet].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        std::size_t index = op + mp + coolist2[i - inlet] + counter1;
        COO_A[index].row = i - inlet;
        COO_A[index].col = Tb[j].ID_2 - inlet;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
      {
        std::size_t index = op + mp + coolist2[i - inlet] + counter1;
        COO_A[index].row = i - inlet;
        COO_A[index].col = Tb[j].ID_2 - para_macro;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        std::size_t index = NA + coolist6[i - inlet];
        if (exists) {
          COO_A[index].col = op + mp + 1;
        } else {
          COO_A[index].col = op + mp;
        }
        COO_A[index].row = i - inlet;
        COO_A[index].val = Pjs[counter].d(0);
        counter++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].C1;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      /*这段被注释掉的程序是为了计算渗透率*/
      // if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
      //   B[i - para_macro] = 0;
      // } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
      //   B[i - para_macro] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      // }
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro] = -F.val();

    COO_A[i - para_macro].row = i - para_macro;
    COO_A[i - para_macro].col = i - para_macro;
    COO_A[i - para_macro].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        std::size_t index = op + mp + coolist2[i - para_macro] + counter1;
        COO_A[index].row = i - para_macro;
        COO_A[index].col = Tb[j].ID_2 - inlet;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
      {
        std::size_t index = op + mp + coolist2[i - para_macro] + counter1;
        COO_A[index].row = i - para_macro;
        COO_A[index].col = Tb[j].ID_2 - para_macro;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        std::size_t index = NA + coolist6[i - para_macro];
        if (exists) {
          COO_A[index].col = op + mp + 1;
        } else {
          COO_A[index].col = op + mp;
        }
        COO_A[index].row = i - para_macro;
        COO_A[index].val = Pjs[counter].d(0);
        counter++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

  // ----------------------------------------------------
  // II. 边界条件组装分派：根据配置类型，执行特定的边界组装
  // ----------------------------------------------------

  // 1. 压力 + 双组分输运 (N_VAR=3) 的 Neumann 边界
  if (dynamic_cast<const LiquidMatrixConfig_2Transport_Neumann*>(&config_)) {
    // 边界方程在矩阵中的起始行： N_VAR * (op + mp)
    const int bc_start_row = op_mp_total * N_VAR;

    // 调用专门的边界组装函数
    assembleNeumannBC_2Transport(dt, mem, mesh_data, Pb, Tb, bc_start_row, op_mp_total, N_ROWS);

  }
  // 2. 压力单变量 (N_VAR=1) 的 Neumann 边界
  else if (dynamic_cast<const LiquidMatrixConfig_PressureOnly_Neumann*>(&config_)) {
    // 边界方程在矩阵中的起始行： N_VAR * (op + mp)
    const int bc_start_row = op_mp_total * N_VAR;

    // 调用专门的边界组装函数
    assembleNeumannBC_POnly(dt, mem, mesh_data, Pb, Tb, bc_start_row, op_mp_total, N_ROWS);
  }

  // Dirichlet 配置和其他非 Neumann 配置会跳过此处的 if/else if 块。
}