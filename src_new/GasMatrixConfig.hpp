#include "MeshInput.hpp"
#include "SystemMatrixConfig.hpp"

class GasMatrixConfig_PressureOnly : public SystemMatrixConfig {
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

  int get_dof_per_pore() const override { return 1; }
};

class GasMatrixConfig_Transport : public SystemMatrixConfig {
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

    // 注意：这里的 coolist 需要从 MeshInput 获取
    // 为简化，我们假设 MeshInput 提供了计算 NA 的方法，或直接将 coolist 移至其 public 成员
    // 为了不大幅修改 MeshInput.hpp，假设 mesh.get_NA() 返回的是单个未知数的基准 NA
    int base_NA = mesh.get_NA();        // 假设这是基于压力方程计算的基础 NA
    int op = mesh.get_op();
    int mp = mesh.get_mp();

    int transport_NA = base_NA + 2 * (op + mp);        // 假设 NA 重新计算，或者从 mesh 拿到 coolist 来计算
    int append_nnz = (mesh.get_inlet() + mesh.get_m_inlet()) * 4 + 2;
    return transport_NA * 5 + append_nnz;        // 这里的 5 和 4 是根据您 CoordinationNumber.cpp 中的分配推测的
  }

  int get_dof_per_pore() const override { return 3; }
};

class GasMatrixConfig_Transport : public SystemMatrixConfig {}