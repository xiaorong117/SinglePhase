#ifndef INITIAL_CONDITION_HPP
#define INITIAL_CONDITION_HPP

class InitialCondition {
 private:
  // 1. 私有化构造函数和析构函数，强制使用单例模式
  InitialCondition();
  ~InitialCondition();

 public:
  // 2. 静态方法获取唯一的实例
  static InitialCondition& getInstance();

  // 3. 禁用复制和赋值
  InitialCondition(const InitialCondition&) = delete;
  InitialCondition& operator=(const InitialCondition&) = delete;

  // 4. 主要功能：设置初始条件
  /**
     * @brief 设置所有求解变量 (压力、速度、浓度等) 的初始值。
     * * 该函数应访问 Memory 单例获取的指针，并根据 MeshInput 提供的尺寸进行循环。
     */
  void setInitialValues();
  // 5. 其他辅助函数 (可选)
  // void initializePressure();
  // void initializeConcentration();
};

#endif        // INITIAL_CONDITION_HPP