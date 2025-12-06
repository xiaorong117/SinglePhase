// Memory.hpp
#include "Globals.hpp"
#ifndef MEMORY_HPP
#define MEMORY_HPP

using namespace std;

class Memory {
 private:
  double *dX, *B;
  // 求解的时间变量 CSR format
  int *ia, *ja;

  // 矩阵的内存空间CSR
  int *rows_offsets, *columns;
  double* values;
  // 矩阵的显存空间CSR
  int *d_csr_offsets, *d_csr_columns;
  double *d_csr_values, *d_M_values;

  // COO format
  int *irn, *jcn;
  double* a;
  Acoo* COO_A;
  // 申请孔喉的动态存储空间
  pore* Pb;
  throat* Tb_in;
  throatmerge* Tb;

  // 关键步骤 1: 将构造函数设为私有，防止外部创建多个实例
  Memory();
  // 关键步骤 2: 将析构函数设为私有 (或者像之前一样保留公有，但单例通常设为私有)
  ~Memory();

 public:
  // 关键步骤 3: 提供静态方法获取唯一的实例
  static Memory& getInstance();
  // 关键步骤 4: 防止复制和赋值（现代 C++ 最佳实践）
  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  // 你可能需要一些公共方法来获取分配的指针，例如：
  double* get_DX() const { return dX; }
  double* get_B() const { return B; }
  int* get_ia() const { return ia; }
  int* get_ja() const { return ja; }
  int* get_rows_offsets() const { return rows_offsets; }
  int* get_columns() const { return columns; }
  double* get_values() const { return values; }
  int* get_d_csr_offsets() const { return d_csr_offsets; }
  int* get_d_csr_columns() const { return d_csr_columns; }
  double* get_d_csr_values() const { return d_csr_values; }
  double* get_d_M_values() const { return d_M_values; }
  int* get_irn() const { return irn; }
  int* get_jcn() const { return jcn; }
  double* get_a() const { return a; }
  Acoo* get_COO_A() const { return COO_A; }
  pore* get_Pb() const { return Pb; }
  throat* get_Tb_in() const { return Tb_in; }
  throatmerge* get_Tb() const { return Tb; }
};

#endif        // MEMORY_HPP