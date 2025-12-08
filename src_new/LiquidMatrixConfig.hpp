#include "MeshInput.hpp"
#include "SystemMatrixConfig.hpp"

class LiquidMatrixConfig_PressureOnly : public SystemMatrixConfig {
 public:
  int get_n_variables(const MeshInput& mesh) const override {
    // 假设：只求解压力 (P)，对应 op + mp 个未知数
    return mesh.get_op() + mesh.get_mp();
  }

  int calculate_nnz(const MeshInput& mesh) const override {
    // 您在 Memory.cpp 中计算的 NA (非零元素) 逻辑
    // 需要 coolist 数据，假设 MeshInput::calculateMeshTopology 已运行
    return mesh.get_NA();        // 假设 NA 是针对压力计算的
  }
};

class LiquidMatrixConfig_PressureOnly_Neumann : public SystemMatrixConfig {
 public:
  int get_n_variables(const MeshInput& mesh) const override {
    // 假设：求解 P ;1个未知数+ 2个边界变量
    return mesh.get_op() + mesh.get_mp() + 2;
  }

  int calculate_nnz(const MeshInput& mesh) const override {
    int append_nnz = (mesh.get_inlet() + mesh.get_m_inlet()) * 2 + 2;
    return mesh.get_NA() + append_nnz;
  }
};

class LiquidMatrixConfig_2Transport_Dirichlet : public SystemMatrixConfig {
 public:
  int get_n_variables(const MeshInput& mesh) const override {
    // 假设：求解 P, C1, C2，对应 (op + mp) * 3 个未知数
    return (mesh.get_op() + mesh.get_mp()) * 3;
  }

  int calculate_nnz(const MeshInput& mesh) const override { return mesh.get_NA() * 5; }
};

class LiquidMatrixConfig_2Transport_Neumann : public SystemMatrixConfig {
 public:
  int get_n_variables(const MeshInput& mesh) const override {
    // 假设：求解 P, C1, C2，对应 (op + mp) * 3 个未知数 + 2个边界变量
    // 对应 Memory.cpp 中 Flag_QIN_trans 的逻辑
    return (mesh.get_op() + mesh.get_mp()) * 3 + 2;
  }

  int calculate_nnz(const MeshInput& mesh) const override {
    // 对应 Memory.cpp 中 Flag_QIN_trans 的逻辑
    // NA = accumulate(...) + op + mp
    // append_nnz = (inlet + m_inlet) * 4 + 2
    // total_nnz = NA * 5 + append_nnz

    int append_nnz = (mesh.get_inlet() + mesh.get_m_inlet()) * 4 + 2;
    return mesh.get_NA() * 5 + append_nnz;
  }
};
