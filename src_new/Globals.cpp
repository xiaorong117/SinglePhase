#include "Globals.hpp"

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
int FLAG = 0;
int number_inlet = 453;
int number_outlet = 1;
std::vector<int> inlet_boundary(number_inlet);

// ------------------------------------
// IV. 命名空间常量和变量的 extern 声明
// ------------------------------------
namespace My_const {
// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
double iters_globa{0};
double pi = 3.1415927;
}        // namespace My_const

namespace Fluid_property {
double gas_vis = 2e-5;                 // 粘度
double D = 9e-9;                       // 扩散系数
double Effect_D = 0.05 * D;            // 微孔中的有效扩散系数
double T_critical{190.564};            // 甲烷的临界温度 190.564K
double P_critical{4.599 * 1e6};        // 甲烷的临界压力 4.599MPa
double Rho_ad{400};                    // kg/m3
double n_max_ad{44.8};                 // kg/m3
double K_langmuir{4e-8};               // Pa^(-1)
double Ds{2.46e-8};                    // m2/s
vector<double> Ds_LIST({8.32e-9, 9.52e-9, 1.14e-8, 1.44e-8, 1.77e-8, 2.10e-8, 2.46e-8});

double D_dispersion{2e-7};
double inlet_co2_mole_frac = 0.9;
double outlet_co2_mole_farc = 0.1;
double MOLE_MASS_CO2{0.046};
double MOLE_MASS_CH4{0.016};

double n_max_ad_co2{45};              // kg/m3
double K_langmuir_co2{1e-7};          // Pa^(-1)
double n_max_ad_ch4{5};               // kg/m3
double K_langmuir_ch4{1.7e-7};        // Pa^(-1)

namespace kong {
double D_dispersion_macro{0.85e-9};        // 237.885 / 100 for 1st,  233.65 / 100 for
                                           // 2nd, 396.055 / 100 for 3rd.
double D_dispersion_micro{0.05 * D_dispersion_macro};
double inlet_C1 = 10;
double outlet_C1 = 0.01;
double inlet_C2 = 10;
double outlet_C2 = 0.01;
double viscosity = 2e-5;
}        // namespace kong
}        // namespace Fluid_property
// ------------------------------------
// V. Porous_media_property 命名空间
// ------------------------------------
namespace Porous_media_property_Hybrid {
double porosity = 0.1;        // 孔隙率
double ko = 1e-15;            // 微孔达西渗透率 m^2
double micro_radius{3.48e-9};
}        // namespace Porous_media_property_Hybrid

namespace Porous_media_property_PNM {
double porosity_HP{0.243};
double porosity_LP{0.081};
double porosity_HP1{0.243};
double porosity_HP2{0.243};
double porosity_LP1{0.081};
double porosity_LP2{0.081};          // 含水
double porosity_clay1{0.081};        // 含水改成0   大孔粘土
double porosity_clay2{0.081};        // 含水改成0
double micro_porosity_HP{0.1};
double micro_porosity_LP{0.1};
double micro_porosity_Clay{0.1};        // 含水改成0,不含水改成porosity

double swww_clay{0};        // 含水改成1，不含水0
double swww_om{0};          // 含水改成 0.5，不含水0
double Sw_OMLP{swww_om};
double Sw_max_OMLP{1};
double a_OMLP = 2e-21 / pow(Sw_max_OMLP, 2);
double K_OM_LP{a_OMLP * pow(Sw_OMLP - Sw_max_OMLP, 2)};

double Sw_om{Sw_OMLP};
double Sw_max_om{1};
double a_om = (700e-21) / pow(Sw_max_om, 2);
double K_OM_HP{a_om * pow(Sw_om - Sw_max_om, 2)};

double Sw_clay{swww_clay};
double Sw_max_clay{1};
double a_clay = 2e-22 / pow(Sw_max_clay, 6);
double K_Clay{a_clay * pow(Sw_clay - Sw_max_clay, 6)};

double L_CLAY{20e-9};
double L_OM_1{1000e-9};
double L_OM_2{20e-9};
}        // namespace Porous_media_property_PNM
// ------------------------------------
// VI. Physical_property 和 Solver_property 命名空间
// ------------------------------------
namespace Physical_property {
double inlet_pre = 100;         // 进口压力 Pa
double outlet_pre = 0;          // 出口压力 Pa
double Temperature{400};        // 温度
double refer_pressure{0};
}        // namespace Physical_property

namespace Solver_property {
double voxel_size = 2e-6;        // 像素尺寸，单位m    5.345e-6 8e-9
double domain_size_cubic = 1770;
int Time_step{0};
int percentage_production_counter{1};
double pyhsic_time{0};
double machine_time{0};
int Flag_eigen{false};
int Flag_Hybrid{false};
int flag = 2;
int flag1 = 2;
int Flag_velocity_bound{false};        // 流速边界
int Flag_species{false};               // 我最开始考虑的，Dirichlet boundary condition 下的transport
int Flag_vector_data{false};           // 输出不考虑进出口的累计全配位文件和只考虑进出口的累计全配位数 vector
int Flag_QIN_trans{false};             // Qin 的Neumann边界条件下的双组分transport模拟
int Flag_QIN_Per{false};               // Qin 的Neumann边界条件下的渗透率模拟

std::string folderPath;
std::string Gfilename("Pe_per");
}        // namespace Solver_property