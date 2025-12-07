#ifndef PHYSICS_MODEL_HPP
#define PHYSICS_MODEL_HPP

#include "Globals.hpp"        // 包含所需的常量
#include "Memory.hpp"

class PhysicsModel {
 public:
  // 纯虚函数 (Pure Virtual Function)
  // "= 0" 表示父类不提供实现，子类必须实现它！这也使得 PhysicsModel 成为抽象类。
  virtual ~PhysicsModel() = default;

  /**
     * @brief 初始时刻 (t=0) 的参数设置
     * 负责计算初始的 Volume, Viscosity, Compressibility 等
     */
  virtual void initialize() = 0;

  /**
     * @brief 时间步循环中的参数更新 (t > 0)
     * 根据当前的压力 (P) 或 浓度 (C) 更新流体属性
     * @param dt 当前时间步长 (可选)
     */
  virtual void updateProperties() = 0;
};

#endif        // PHYSICS_MODEL_HPP