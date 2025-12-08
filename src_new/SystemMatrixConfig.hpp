#ifndef SYSTEM_MATRIX_CONFIG_HPP
#define SYSTEM_MATRIX_CONFIG_HPP

class MeshInput;        // 前置声明

/**
 * @brief 系统矩阵配置的抽象基类 (策略模式)
 * 封装了不同物理模型下求解矩阵的维度和NNZ计算逻辑。
 */
class SystemMatrixConfig {
 public:
  virtual ~SystemMatrixConfig() = default;

  // 1. 获取系统变量总数 (行数/未知数数量)
  virtual int get_n_variables(const MeshInput& mesh) const = 0;

  // 2. 估算非零元素总数 (NNZ)
  virtual int calculate_nnz(const MeshInput& mesh) const = 0;
};

#endif        // SYSTEM_MATRIX_CONFIG_HPP