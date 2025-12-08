// JacobianAssembler_LinearEquation.hpp

#ifndef JACOBIAN_ASSEMBLER_LINEAR_EQUATION_HPP
#define JACOBIAN_ASSEMBLER_LINEAR_EQUATION_HPP

#include "JacobianAssembler.hpp"        // 继承自 JacobianStrategy
#include "MeshInput.hpp"
#include "SystemMatrixConfig.hpp"

/**
 * @brief 抽象基类：处理线性方程组的雅可比组装策略。
 * 核心特征：直接计算矩阵 A 和向量 B，不使用 AD。
 */
class JacobianAssembler_LinearEquation : public JacobianStrategy {
 protected:
  const SystemMatrixConfig& config_;        // 配置策略

  /**
   * @brief 辅助函数：推断每个节点变量数 N_VAR。
   */
  virtual int deduce_n_var_per_node(const SystemMatrixConfig& config) const = 0;

 public:
  // 构造函数：初始化通用字段
  JacobianAssembler_LinearEquation(const SystemMatrixConfig& config) : config_(config){};

  virtual ~JacobianAssembler_LinearEquation() = default;

  // 核心组装函数仍然是纯虚函数，由具体子类（如 SolidMechanicsAssembler）实现
  void assemble(double dt) override = 0;
};

#endif        // JACOBIAN_ASSEMBLER_LINEAR_EQUATION_HPP