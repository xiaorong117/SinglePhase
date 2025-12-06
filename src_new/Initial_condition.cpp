#include "Initial_condition.hpp"
#include <omp.h>        // For omp_get_wtime
#include <fstream>
#include "Memory.hpp"           // 包含 Memory
#include "MeshInput.hpp"        // 包含 MeshInput

using namespace Physical_property;
using namespace Fluid_property;
using namespace Solver_property;
using namespace My_const;
// ----------------------------------------------------
// 1. 实现 getInstance()
// ----------------------------------------------------
InitialCondition& InitialCondition::getInstance() {
  // 保证线程安全和懒汉式初始化
  static InitialCondition instance;
  return instance;
}        // namespace InitialCondition::getInstance()

InitialCondition::InitialCondition() {};

void InitialCondition::setInitialValues() {
  double start = omp_get_wtime();

  // 访问单例获取尺寸和指针
  MeshInput& mesh_data = MeshInput::getInstance();
  Memory& mem = Memory::getInstance();

  // 获取尺寸
  const int pn = mesh_data.get_pn();
  const int macro_n = mesh_data.get_macro_n();
  const int outlet = mesh_data.get_outlet();
  const int m_outlet = mesh_data.get_m_outlet();
  const int inlet = mesh_data.get_inlet();
  const int m_inlet = mesh_data.get_m_inlet();
  throatmerge* Tb = mem.get_Tb();

  // 获取数据指针
  pore* Pb = mem.get_Pb();

  for (int i = 0; i < pn; i++) {
    Pb[i].pressure = inlet_pre;        //- double(double(i) / double(pn) * 100)
    Pb[i].pressure_old = Pb[i].pressure;
  }
  for (int i = macro_n - outlet; i < macro_n; i++) {
    Pb[i].pressure = outlet_pre;
    Pb[i].pressure_old = outlet_pre;
  }
  for (int i = pn - m_outlet; i < pn; i++) {
    Pb[i].pressure = outlet_pre;
    Pb[i].pressure_old = outlet_pre;
  }

  // for (int i = 0; i < inlet; i++) {
  //   Pb[i].pressure = inlet_pre;
  //   Pb[i].pressure_old = inlet_pre;
  // }

  // for (int i = macro_n; i < macro_n + m_inlet; i++) {
  //   Pb[i].pressure = inlet_pre;
  //   Pb[i].pressure_old = inlet_pre;
  // }

  for (int i = 0; i < pn; i++) {
    Pb[i].C1 = kong::outlet_C1;        //- double(double(i) / double(pn) * 100)
    Pb[i].C1_old = Pb[i].C1;
    Pb[i].C2 = kong::outlet_C2;        //- double(double(i) / double(pn) * 100)
    Pb[i].C2_old = Pb[i].C2;
  }

  // for (int i = 0; i < inlet; i++) {
  //   Pb[i].C1 = kong::inlet_C1;        //- double(double(i) / double(pn) * 100)
  //   Pb[i].C1_old = Pb[i].C1;
  //   Pb[i].C2 = kong::inlet_C2;        //- double(double(i) / double(pn) * 100)
  //   Pb[i].C2_old = Pb[i].C2;
  // }

  // for (int i = macro_n; i < macro_n + m_inlet; i++) {
  //   Pb[i].C1 = kong::inlet_C1;        //- double(double(i) / double(pn) * 100)
  //   Pb[i].C1_old = Pb[i].C1;
  //   Pb[i].C2 = kong::inlet_C2;        //- double(double(i) / double(pn) * 100)
  //   Pb[i].C2_old = Pb[i].C2;
  // }

  // for (int i = macro_n - outlet; i < macro_n; i++) {
  //   Pb[i].C1 = kong::outlet_C1;
  //   Pb[i].C1_old = Pb[i].C1;
  //   Pb[i].C2 = kong::outlet_C2;
  //   Pb[i].C2_old = Pb[i].C2;
  // }
  // for (int i = pn - m_outlet; i < pn; i++) {
  //   Pb[i].C1 = kong::outlet_C1;
  //   Pb[i].C1_old = Pb[i].C1;
  //   Pb[i].C2 = kong::outlet_C2;
  //   Pb[i].C2_old = Pb[i].C2;
  // }

  ifstream inlet_coo1("filtered_inlet_coo.txt", ios::in);
  if (!inlet_coo1.is_open()) {
    cout << "inlet_coo1 file not found!" << endl;
    abort();
  }

  for (size_t i = 0; i < number_inlet; i++) {
    double x, y, z, r;
    int id, type;
    inlet_coo1 >> x >> y >> z >> id >> r >> type;
    inlet_boundary[i] = id;
  }
  inlet_coo1.close();

  std::sort(inlet_boundary.begin(), inlet_boundary.end());

  /*我只是想在这个地方更新一下边界条件*/
  if (Flag_velocity_bound == true)
  /*如果考虑Neumann边界条件的话，边界上的压力给初始值*/ {
    for (int i = 0; i < inlet; i++) {
      bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
      for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
        reverse_mode<double> Pi, Wi;
        reverse_mode<double>* Pjs;
        reverse_mode<double>* Wjs;
        double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
        double Aij = pi * pow(Tb[j].Radiu, 2);
        if (exists) {
          double vij = 1.29 * 0.01 / 60;                                       // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);        // 更新孔隙压力
          Pb[i].pressure_old = Pb[i].pressure;
        } else {
          double vij = 1.04 * 0.01 / 60;                                       // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);        // 更新孔隙压力
          Pb[i].pressure_old = Pb[i].pressure;
        }
      }
    }
    for (int i = macro_n; i < macro_n + m_inlet; i++) {
      bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
      for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
        reverse_mode<double> Pi, Wi;
        reverse_mode<double>* Pjs;
        reverse_mode<double>* Wjs;
        double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
        double Aij = pi * pow(Tb[j].Radiu, 2);
        if (exists) {
          double vij = 1.29 * 0.01 / 60;                                       // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);        // 更新孔隙压力
          Pb[i].pressure_old = Pb[i].pressure;
        } else {
          double vij = 1.04 * 0.01 / 60;                                       // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);        // 更新孔隙压力
          Pb[i].pressure_old = Pb[i].pressure;
        }
      }
    }
  }

  if (Flag_QIN_trans == true)
  /*加上了Qin说的Neumann边界条件之后，变量数量改变了，transport。*/
  {
    for (int i = 0; i < inlet; i++) {
      bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
      for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
        if (exists) {
          Pb[i].C1 = 0;        // 更新孔隙压力
          Pb[i].C1_old = Pb[i].C1;
          Pb[i].C2 = kong::inlet_C2;
          Pb[i].C2_old = Pb[i].C2;
        } else {
          Pb[i].C1 = kong::inlet_C1;        // 更新孔隙压力
          Pb[i].C1_old = Pb[i].C1;
          Pb[i].C2 = 0;
          Pb[i].C2_old = Pb[i].C2;
        }
      }
    }
    for (int i = macro_n; i < macro_n + m_inlet; i++) {
      bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
      for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
        if (exists) {
          Pb[i].C1 = 0;        // 更新孔隙压力
          Pb[i].C1_old = Pb[i].C1;
          Pb[i].C2 = kong::inlet_C2;
          Pb[i].C2_old = Pb[i].C2;
        } else {
          Pb[i].C1 = kong::inlet_C1;        // 更新孔隙压力
          Pb[i].C1_old = Pb[i].C1;
          Pb[i].C2 = 0;
          Pb[i].C2_old = Pb[i].C2;
        }
      }
    }
  }
  /*我只是想在这个地方更新一下边界条件*/
  double end = omp_get_wtime();
  printf("initial_condition start = %.16g\tend = %.16g\tdiff = %.16g\n", start, end, end - start);
};