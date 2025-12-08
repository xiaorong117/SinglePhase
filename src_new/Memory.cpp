#include "Memory.hpp"
#include "Globals.hpp"
#include "MeshInput.hpp"
#include "SelfDefinedFunctions.hpp"
// 补充的必需标准库头文件
#include <cassert>         // For assert()
#include <cstdlib>         // For abort()
#include <fstream>         // For std::ifstream, std::ios
#include <iostream>        // For std::cout, std::endl, std::cerr
#include <numeric>
#include <sstream>        // For std::istringstream
#include <string>         // For std::string, std::string::npos
#include <vector>         // For std::vector (虽然在您的代码片段中已使用 using namespace std;, 但显式包含是好习惯)
using namespace Solver_property;
using namespace std;

// ------------------------------------
// Memory 类的构造函数定义
// ------------------------------------
Memory& Memory::getInstance() {
  // 关键点：使用静态局部变量
  // 1. 保证 Memory 实例只创建一次。
  // 2. 第一次调用此函数时，会自动调用 Memory::Memory() 构造函数完成内存分配。
  // 3. 在程序退出时，该静态对象会自动调用 Memory::~Memory() 析构函数完成内存释放。
  static Memory instance;
  return instance;
}

void Memory::allocateSolverMatrixMemory(const SystemMatrixConfig& config) {
  // 1. 获取 MeshInput (用于获取 pn, tn, inlet, op, mp 等)
  MeshInput& mesh_data = MeshInput::getInstance();

  // 2. 根据策略对象获取所需的总变量数 (矩阵行/列数)
  const int n_variables = config.get_n_variables(mesh_data);

  // 3. 根据策略对象计算所需的非零元素总数 (NNZ)
  const int total_nnz = config.calculate_nnz(mesh_data);

  std::cout << "Allocating solver matrix memory..." << std::endl;
  std::cout << "  - N_Variables (Rows/Cols): " << n_variables << std::endl;
  std::cout << "  - NNZ (Non-Zero Entries):  " << total_nnz << std::endl;

  // 4. 动态分配内存
  // 删除旧内存 (防止内存泄漏)
  // ... (略去现有代码中的 delete[] 逻辑)

  // 分配 dX 和 B 向量 (大小为 N_Variables)
  dX = new double[n_variables];
  B = new double[n_variables];

  // 分配 CSR/COO 矩阵所需的内存
  // ia (行偏移) 大小为 N_Variables + 1
  ia = new int[n_variables + 1];

  // ja (列索引), a (值), COO_A (COO格式) 大小为 NNZ
  ja = new int[total_nnz];
  a = new double[total_nnz];
  COO_A = new Acoo[total_nnz];

  // ... 其他分配逻辑 (如 d_csr_offsets 等)
  std::cout << "Solver matrix memory allocation complete." << std::endl;
};

Memory::Memory() {
  // 初始化所有指针为 nullptr (如果你没有在声明时初始化)
  dX = B = nullptr;
  ia = ja = nullptr;
  rows_offsets = columns = nullptr;
  values = nullptr;
  d_csr_offsets = d_csr_columns = nullptr;
  d_csr_values = d_M_values = nullptr;
  irn = jcn = nullptr;
  a = nullptr;
  COO_A = nullptr;
  Pb = nullptr;
  Tb_in = nullptr;
  Tb = nullptr;

  MeshInput& mesh_data = MeshInput::getInstance();
  const int pn = mesh_data.get_pn();
  const int tn = mesh_data.get_tn();

  Pb = new pore[pn];
  Tb_in = new throat[2 * tn];
  Tb = new throatmerge[2 * tn];
}
Memory::~Memory() {
  // 逐一安全释放所有指针，使用 delete[]
  delete[] dX;
  delete[] B;
  delete[] ia;
  delete[] ja;

  delete[] rows_offsets;
  delete[] columns;
  delete[] values;

  delete[] d_csr_offsets;
  delete[] d_csr_columns;
  delete[] d_csr_values;
  delete[] d_M_values;

  delete[] irn;
  delete[] jcn;
  delete[] a;
  delete[] COO_A;        // 如果 Acoo 是一个数组，用 delete[]

  delete[] Pb;
  delete[] Tb_in;
  delete[] Tb;

  // C++ 保证在 delete[] nullptr 是安全的，所以如果分配失败，
  // 只要你在构造函数开头初始化了 nullptr，这里就没问题。
}