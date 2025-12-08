// JacobianAssembler_NonLinearEquation.hpp

#ifndef JACOBIAN_ASSEMBLER_NON_LINEAR_EQUATION_HPP
#define JACOBIAN_ASSEMBLER_NON_LINEAR_EQUATION_HPP

#include "JacobianAssembler.hpp"        // 继承自 JacobianStrategy
#include "MeshInput.hpp"
#include "SystemMatrixConfig.hpp"

/**
 * @brief 抽象基类：处理非线性方程组的雅可比组装策略。
 * 核心特征：必须使用自动微分 (AD) 技术计算残差和雅可比。
 */
class JacobianAssembler_NonLinearEquation : public JacobianStrategy {
 protected:
  const SystemMatrixConfig& config_;        // 配置策略

  /**
   * @brief 辅助函数：推断每个节点变量数 N_VAR。
   * @details 交由子类实现，以处理具体的 Liquid/Gas/Solid 配置。
   */
  virtual int deduce_n_var_per_node(const SystemMatrixConfig& config) const = 0;

 public:
  // 构造函数：初始化通用字段
  JacobianAssembler_NonLinearEquation(const SystemMatrixConfig& config) : config_(config) {}

  virtual ~JacobianAssembler_NonLinearEquation() = default;

  // 核心组装函数仍然是纯虚函数，由 LiquidJacobianAssembler 等具体类实现
  void assemble(double dt) override = 0;

  // 可以在此放置通用的 AD 变量初始化、内存清理等辅助方法
  // 例如：void initializeADVariables(int i, reverse_mode<double>*& current_vars, ...);
};

#endif        // JACOBIAN_ASSEMBLER_NON_LINEAR_EQUATION_HPP