#ifndef JACOBIAN_ASSEMBLER_HPP
#define JACOBIAN_ASSEMBLER_HPP

// 抽象策略接口
class JacobianStrategy {
 public:
  // 必须有一个虚析构函数，以确保通过基类指针删除子类对象时能正确调用子类析构函数
  virtual ~JacobianStrategy() = default;

  /**
   * @brief 组装雅可比矩阵和右端项向量。
   * @param dt 时间步长。
   */
  virtual void assemble(double dt) = 0;        // 纯虚函数，子类必须实现
};

#endif        // JACOBIAN_ASSEMBLER_HPP