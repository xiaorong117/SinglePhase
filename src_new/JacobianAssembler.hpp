#ifndef JACOBIAN_ASSEMBLER_HPP
#define JACOBIAN_ASSEMBLER_HPP

// 抽象策略接口
class JacobianStrategy {
 public:
  virtual ~JacobianStrategy() = default;

  /**
     * @brief 组装雅可比矩阵和右端项向量。
     * * @param dt 时间步长 (如果模型是瞬态的)
     */
  virtual void assemble(double dt) = 0;
};

#endif        // JACOBIAN_ASSEMBLER_HPP