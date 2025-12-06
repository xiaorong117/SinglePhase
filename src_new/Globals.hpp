// Globals.hpp

#ifndef GLOBALS_HPP
#define GLOBALS_HPP

// ------------------------------------
// 必需的头文件和类型别名
// ------------------------------------
#include <algorithm>        // 用于 std::min
#include <cmath>            // 用于 std::pow, std::min 等（虽然主要在实现中用，但这里声明了带实现逻辑的常量）
#include <string>
#include <vector>
#include "badiff.h"        // 假设这些是 AD 库
#include "fadiff.h"

// 假设这些库类型也需要在其他地方使用
template <typename T>
using reverse_mode = B<T>;

// ------------------------------------
// I. 宏定义 (必须保留在头文件中)
// ------------------------------------
// 宏定义不使用 extern，直接保留在头文件中
#define OMP_PARA 20
#define PE "kong_pe_1"

// ------------------------------------
// II. 结构体定义 (直接在头文件中定义，因为它们不涉及分离实现)
// ------------------------------------
typedef struct coo {
  int row, col;
  double val;
} Acoo;

struct pore {
  double X{0};
  double Y{0};
  double Z{0};
  double Radiu{0};
  int Half_coord{0};
  int half_accum{0};
  int full_coord{0};
  int full_accum{0};
  int full_coord_ori{0};
  int full_accum_ori{0};
  int type{0};
  int main_gate{0};
  double pressure{0};
  double pressure_old{0};
  double mole_frac_co2{0};
  double mole_frac_co2_old{0};
  double C1{0};
  double C2{0};
  double C1_old{0};
  double C2_old{0};
  double volume{0};
  double compre{0};
  double compre_old{0};
  double visco{0};
  double visco_old{0};
  double km{0};
  double porosity{0};
  double radius_micro{0};
};

struct throat {
  int ID_1{0};
  int ID_2{0};
  int n_direction{0};
  double Radiu{0};
  double Length{0};
  double Conductivity{0};
  double Surface_diff_conduc{0};
  double Knusen{0};
  double Slip{0};
  double center_x{0};
  double center_y{0};
  double center_z{0};
  double surface_diff_co2{0};
  double surface_diff_methane{0};
  double dispersion_coe_co2{0};
  int main_free{0};
};

struct throatmerge {
  int ID_1{0};
  int ID_2{0};
  double Radiu{0};
  double Conductivity{0};
  double Surface_diff_conduc{0};
  double Knusen{0};
  double Slip{0};
  double surface_diff_co2{0};
  double surface_diff_methane{0};
  double dispersion_coe_co2{0};
  int main_free{0};
  int main_surface{0};
  double Pore_1{0};
  double Pore_2{0};
};

// ------------------------------------
// III. 全局变量的 extern 声明
// ------------------------------------

// 使用 extern 声明，告诉编译器这些变量在 Globals.cpp 中有且仅有一个定义
extern int FLAG;
extern int number_inlet;
extern int number_outlet;
extern std::vector<int> inlet_boundary;

// ------------------------------------
// IV. 命名空间常量和变量的 extern 声明
// ------------------------------------

namespace My_const {
// 常量可以带 extern const，但如果 CLOCKS_PER_SECOND 打算作为 const 放在头文件，
// 则不需要 extern，这里假设为 extern 变量
extern const double CLOCKS_PER_SECOND;
extern double iters_globa;
extern double pi;
}        // namespace My_const

namespace Fluid_property {
extern double gas_vis;                     // 粘度
extern double D;                           // 扩散系数
extern double Effect_D;                    // 微孔中的有效扩散系数
extern double T_critical;                  // 甲烷的临界温度 190.564K
extern double P_critical;                  // 甲烷的临界压力 4.599MPa
extern double Rho_ad;                      // kg/m3
extern double n_max_ad;                    // kg/m3
extern double K_langmuir;                  // Pa^(-1)
extern double Ds;                          // m2/s
extern std::vector<double> Ds_LIST;        // 注意 vector 也要 extern
extern double D_dispersion;
extern double inlet_co2_mole_frac;
extern double outlet_co2_mole_farc;
extern double MOLE_MASS_CO2;
extern double MOLE_MASS_CH4;

extern double n_max_ad_co2;          // kg/m3
extern double K_langmuir_co2;        // Pa^(-1)
extern double n_max_ad_ch4;          // kg/m3
extern double K_langmuir_ch4;        // Pa^(-1)

namespace kong {
extern double D_dispersion_macro;
extern double D_dispersion_micro;
extern double inlet_C1;
extern double outlet_C1;
extern double inlet_C2;
extern double outlet_C2;
extern double viscosity;
}        // namespace kong
}        // namespace Fluid_property

// ------------------------------------
// V. Porous_media_property 命名空间
// ------------------------------------

namespace Porous_media_property_Hybrid {
extern double porosity;
extern double ko;
extern double micro_radius;
}        // namespace Porous_media_property_Hybrid

namespace Porous_media_property_PNM {
extern double porosity_HP;
extern double porosity_LP;
extern double porosity_HP1;
extern double porosity_HP2;
extern double porosity_LP1;
extern double porosity_LP2;          // 含水
extern double porosity_clay1;        // 含水改成0   大孔粘土
extern double porosity_clay2;
extern double micro_porosity_HP;
extern double micro_porosity_LP;
extern double micro_porosity_Clay;

extern double swww_clay;        // 含水改成1，不含水0
extern double swww_om;          // 含水改成 0.5，不含水0
extern double Sw_OMLP;
extern double Sw_max_OMLP;
extern double a_OMLP;
extern double K_OM_LP;

extern double Sw_om;
extern double Sw_max_om;
extern double a_om;
extern double K_OM_HP;

extern double Sw_clay;
extern double Sw_max_clay;
extern double a_clay;
extern double K_Clay;

extern double L_CLAY;
extern double L_OM_1;
extern double L_OM_2;
}        // namespace Porous_media_property_PNM

// ------------------------------------
// VI. Physical_property 和 Solver_property 命名空间
// ------------------------------------

namespace Physical_property {
extern double inlet_pre;
extern double outlet_pre;
extern double Temperature;
extern double refer_pressure;
}        // namespace Physical_property

namespace Solver_property {
extern double voxel_size;
extern double domain_size_cubic;
extern int Time_step;
extern int percentage_production_counter;
extern double pyhsic_time;
extern double machine_time;
extern int Flag_eigen;
extern int Flag_Hybrid;
extern int flag;
extern int flag1;
extern int Flag_velocity_bound;        // 流速边界
extern int Flag_species;               // 我最开始考虑的，Dirichlet boundary condition 下的transport
extern int Flag_vector_data;           // 输出不考虑进出口的累计全配位文件和只考虑进出口的累计全配位数 vector
extern int Flag_QIN_trans;             // Qin 的Neumann边界条件下的双组分transport模拟
extern int Flag_QIN_Per;               // Qin 的Neumann边界条件下的渗透率模拟

extern std::string folderPath;
extern std::string Gfilename;

}        // namespace Solver_property

#endif        // GLOBALS_HPP