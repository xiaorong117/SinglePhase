#ifndef MESHINPUT_HPP
#define MESHINPUT_HPP

#include <iostream>
#include <string>
#include <vector>

class MeshInput {
 private:
  // 1. **【修正】** 将构造函数和析构函数移到 private 区域
  MeshInput();
  ~MeshInput();

  // 网格尺寸变量 (保持 private 封装)
  int pn;        // 孔隙节点总数
  int tn;        // 喉道总数
  int inlet, outlet, m_inlet, m_outlet, op, mp;
  int macro_n;
  int micro_n;
  int para_macro;
  int NA;

  vector<int> coolist;
  vector<int> coolist2;
  vector<int> coolist5;
  vector<int> coolist6;

 public:
  // 2. **【保持】** 静态方法获取唯一的实例
  static MeshInput& getInstance();

  // 3. **【保持】** 防止复制和赋值（现代 C++ 最佳实践）
  MeshInput(const MeshInput&) = delete;
  MeshInput& operator=(const MeshInput&) = delete;

  // 4. **【保持】** 公共方法用于网格数据填充
  void loadMeshStructures();           // 用于执行复杂的网格数据填充
  void calculateMeshTopology();        // <<< 新增：用于执行喉道合并和配位数计算

  // 5. **【保持】** 公共 Getter 方法
  int get_pn() const { return pn; }
  int get_tn() const { return tn; }
  int get_inlet() const { return inlet; }
  int get_outlet() const { return outlet; }
  int get_m_inlet() const { return m_inlet; }
  int get_m_outlet() const { return m_outlet; }
  int get_op() const { return op; }
  int get_mp() const { return mp; }
  int get_macro_n() const { return macro_n; }
  int get_micro_n() const { return micro_n; }
  int get_para_macro() const { return para_macro; }
  int get_NA() const { return NA; }
  const std::vector<int>& get_coolist() const { return coolist; }
  const std::vector<int>& get_coolist2() const { return coolist2; }
  const std::vector<int>& get_coolist5() const { return coolist5; }
  const std::vector<int>& get_coolist6() const { return coolist6; }
};

#endif        // MESHINPUT_HPP