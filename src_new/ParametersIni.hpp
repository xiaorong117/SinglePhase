#ifndef PARAMETERSINI_HPP
#define PARAMETERSINI_HPP

class ParametersIni {
 private:
  // 1. 私有化构造函数和析构函数
  ParametersIni();
  ~ParametersIni();

 public:
  // 2. 静态方法获取唯一的实例
  static ParametersIni& getInstance();

  // 3. 禁用复制和赋值
  ParametersIni(const ParametersIni&) = delete;
  ParametersIni& operator=(const ParametersIni&) = delete;

  // 4. 核心功能：更新物理参数到 Memory 中的数据结构
  /**
     * @brief 访问 MeshInput 和 Memory，将物理和流体参数赋给 Pb 和 Tb_in 数组。
     */
  void initializePhysicalParameters();

 private:
  // 5. 辅助方法：将不同的初始化逻辑模块化
  void initializePoreParameters();
  void initializeThroatParameters();
  void calculatePoreVolumeAndFluidProps();
  // 可以在这里添加其他辅助方法，如读取配置文件
};

#endif        // PARAMETERSINI_HPP