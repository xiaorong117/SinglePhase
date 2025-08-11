#include <dirent.h>
#include <math.h>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>  // 函数所在头文件
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"

// For gsl
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_roots.h>

// For AMGX
#include <amgx_c.h>
#include <amgx_config.h>

/* Using updated (v2) interfaces to cublas usparseSparseToDense*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>       // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
// extern "C" {
//   #include "mkl.h"
//   }

#include "badiff.h"
#include "fadiff.h"

#define OMP_PARA 20
#define PE "kong_pe_1";
using namespace std;
using namespace std::chrono;
using namespace fadbad;

template <typename T>
using reverse_mode = B<T>;
static int FLAG = 0;
std::vector<int> inlet_boundary(1259);
// 输出vector到文件
void writeVectorToFile(const std::vector<int>& vec, const std::string& filename) {
  std::ofstream outFile(filename);
  if (!outFile) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return;
  }

  for (const auto& num : vec) {
    outFile << num << " ";  // 用空格分隔数字
  }

  outFile.close();
  std::cout << "vector已写入文件: " << filename << std::endl;
}

std::vector<int> readVectorFromFile(const std::string& filename) {
  std::vector<int> vec;
  std::ifstream inFile(filename);
  if (!inFile) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return vec;  // 返回空vector
  }

  int num;
  while (inFile >> num) {
    vec.push_back(num);
  }

  inFile.close();
  std::cout << "vector已从文件读取: " << filename << std::endl;
  return vec;
}

namespace My_const {
// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
double iters_globa{0};
double pi = 3.1415927;
}  // namespace My_const

namespace Fluid_property {
double gas_vis = 2e-5;           // 粘度
double D = 9e-9;                 // 扩散系数
double Effect_D = 0.05 * D;      // 微孔中的有效扩散系数
double T_critical{190.564};      // 甲烷的临界温度 190.564K
double P_critical{4.599 * 1e6};  // 甲烷的临界压力 4.599MPa
double Rho_ad{400};              // kg/m3
double n_max_ad{44.8};           // kg/m3
double K_langmuir{4e-8};         // Pa^(-1)
double Ds{2.46e-8};              // m2/s
vector<double> Ds_LIST({8.32e-9, 9.52e-9, 1.14e-8, 1.44e-8, 1.77e-8, 2.10e-8, 2.46e-8});

double D_dispersion{2e-7};
double inlet_co2_mole_frac = 0.9;
double outlet_co2_mole_farc = 0.1;
double MOLE_MASS_CO2{0.046};
double MOLE_MASS_CH4{0.016};

double n_max_ad_co2{45};        // kg/m3
double K_langmuir_co2{1e-7};    // Pa^(-1)
double n_max_ad_ch4{5};         // kg/m3
double K_langmuir_ch4{1.7e-7};  // Pa^(-1)

namespace kong {
double D_dispersion_macro{0.85e-9};  // 237.885 / 100 for 1st,  233.65 / 100 for
                                     // 2nd, 396.055 / 100 for 3rd.
double D_dispersion_micro{0.05 * D_dispersion_macro};
double inlet_co2_mole_frac = 1;
double outlet_co2_mole_farc = 0;
double viscosity = 2e-5;
}  // namespace kong
}  // namespace Fluid_property

namespace Porous_media_property_Hybrid {
double porosity = 0.1;  // 孔隙率
double ko = 1e-15;      // 微孔达西渗透率 m^2
double micro_radius{3.48e-9};
}  // namespace Porous_media_property_Hybrid

namespace Porous_media_property_PNM {
double porosity_HP{0.243};
double porosity_LP{0.081};
double porosity_HP1{0.243};
double porosity_HP2{0.243};
double porosity_LP1{0.081};
double porosity_LP2{0.081};    // 含水
double porosity_clay1{0.081};  // 含水改成0   大孔粘土
double porosity_clay2{0.081};  // 含水改成0
double micro_porosity_HP{0.1};
double micro_porosity_LP{0.1};
double micro_porosity_Clay{0.1};  // 含水改成0,不含水改成porosity

double swww_clay{0};  // 含水改成1，不含水0
double swww_om{0};    // 含水改成 0.5，不含水0
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
}  // namespace Porous_media_property_PNM

namespace Physical_property {
double inlet_pre = 100;   // 进口压力 Pa
double outlet_pre = 0;    // 出口压力 Pa
double Temperature{400};  // 温度
double refer_pressure{0};
}  // namespace Physical_property

namespace Solver_property {
double voxel_size = 2e-6;  // 像素尺寸，单位m    5.345e-6 8e-9
double domain_size_cubic = 2000;
int Time_step{0};
int percentage_production_counter{1};
double pyhsic_time{0};
double machine_time{0};
int Flag_eigen{true};
int Flag_Hybrid{true};
int flag = 2;
int flag1 = 2;

std::string folderPath;
std::string Gfilename("Pe_100");

int pn = 1;  // 505050不联通 sample3  r=2
int tn = 1;
int inlet = 1, outlet = 1, m_inlet = 1, m_outlet = 1, op = 1, mp = 1;

int macro_n = inlet + op + outlet;
int micro_n = m_inlet + mp + m_outlet;
int para_macro = inlet + outlet + m_inlet;
int NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

double getmax_2(double a, double b) {
  return a > b ? a : b;
}

double getmax_3(double a, double b, double c) {
  double temp = getmax_2(a, b);
  temp = getmax_2(temp, c);
  return temp;
}

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
};
std::vector<std::string> getFilesInFolder(const std::string& folderPath) {
  std::vector<std::string> fileList;

  DIR* dir;
  struct dirent* entry;

  // 打开文件夹
  dir = opendir(folderPath.c_str());
  if (dir == nullptr) {
    return fileList;
  }

  // 读取文件夹中的文件
  while ((entry = readdir(dir)) != nullptr) {
    // 忽略当前目录和上级目录
    if (std::string(entry->d_name) == "." || std::string(entry->d_name) == "..") {
      continue;
    }

    // 将文件名添加到列表中
    fileList.push_back(entry->d_name);
  }

  // 关闭文件夹
  closedir(dir);

  return fileList;
}

typedef struct coo {
  int row, col;
  double val;
} Acoo;

int sort_by_row(const void* a, const void* b) {
  if (((Acoo*)a)->row != ((Acoo*)b)->row) {
    return ((Acoo*)a)->row > ((Acoo*)b)->row;
  } else {
    return ((Acoo*)a)->col > ((Acoo*)b)->col;
  }
}
}  // namespace Solver_property

using namespace My_const;
using namespace Fluid_property;
using namespace Porous_media_property_Hybrid;
// using namespace Porous_media_property_PNM;
using namespace Physical_property;
using namespace Solver_property;

class PNMsolver  // 定义类
{
 public:
  PNMsolver(){};
  void AMGX_solver_CO2_methane();  // 混合模型二氧化碳驱替甲烷
  void AMGX_solver_C_kong_PNM();   // kong

 private:
  double *dX, *B;
  // 求解的时间变量 CSR format
  int *ia, *ja;
  // COO format
  int *irn, *jcn;
  double* a;
  Acoo* COO_A;
  // 申请孔喉的动态存储空间
  pore* Pb;
  throat* Tb_in;
  throatmerge* Tb;
  vector<int> coolist;
  vector<int> coolist2;
  vector<vector<int>> coolist3;
  vector<vector<double>> coolist4;

  double error;
  int time_step = Time_step;
  double time_all = pyhsic_time;
  double dt = 1e-2;
  double dt2 = 1e-8;  // 与dt初值相同，用于输出结果文件
  double Q_outlet_macro{0};
  double Q_outlet_free_micro{0};
  double Q_outlet_ad_micro{0};
  double Q_outlet_REV{0};

  double free_macro_loss{0};
  double free_micro_loss{0};
  double ad_micro_loss{0};

  double clay_loss{0};
  double fracture_loss{0};
  double OM_HP_free_loss{0};
  double OM_LP_free_loss{0};
  double OM_HP_ad_loss{0};
  double OM_LP_ad_loss{0};

  double total_p{0};  // total gas content in research domian
  double total_macro{0};
  double total_micro_free{0};
  double total_micro_ad{0};

  double total_clay{0};
  double total_fracture{0};
  double total_OM_HP_free{0};
  double total_OM_LP_free{0};
  double total_OM_HP_ad{0};
  double total_OM_LP_ad{0};

  double norm_inf = 0;
  double eps = 1e-5;      // set residual for dx
  double eps_per = 1e-3;  // set residual for dx

  int iterations_number = 0;

  void memory();  // 动态分配存储器

  void initial_condition();
  void initial_condition(int i);  // 断电继续算

  void Paramentinput();             // 孔喉数据导入函数声明
  void Paramentinput(int i);        // 微孔非均匀文件读取
  void para_cal();                  // 喉道长度等相关参数计算
  void para_cal_in_newton();        // 在牛顿迭代中计算 克努森数
  void para_cal(double);            // 喉道长度等相关参数计算
  void para_cal_in_newton(double);  // 在牛顿迭代中计算 克努森数

  void para_cal_co2_methane();  // 喉道长度等相关参数计算
  reverse_mode<double> conductivity_sur_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_bulk_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_bulk(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_DISPERSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_mehane_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  vector<reverse_mode<double>> ex_adsorption(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int pore_id);
  vector<double> ex_adsorption(double Pi, double Wi, int pore_id);  // 压缩系数
  vector<reverse_mode<double>> ex_adsorption_pro(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int pore_id);
  vector<double> ex_adsorption_pro(double Pi, double Wi,
                                   int pore_id);  // 压缩系数

  reverse_mode<double> conductivity_bulk_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_DISPERSION_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);

  void para_cal_kong();

  double compre(double pressure);  // 压缩系数
  double visco(double pressure, double z, double T);
  void Function_DS(double pressure);
  double Function_Slip(double knusen);
  double Function_Slip_clay(double knusen);

  void CO2_methane_matrix();
  void kong_matrix();
  void CSR2COO();
  void Matrix_COO2CSR();

  reverse_mode<double> func(reverse_mode<double>& Pi, reverse_mode<double>*& Pjs, int num);
  reverse_mode<double> func_BULK_PHASE_FLOW_in_macro_produc(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_BULK_PHASE_FLOW_in_micro_produc(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_BULK_PHASE_FLOW_in_macro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_BULK_PHASE_FLOW_in_micro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_TRANSPORT_FLOW_in_macro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_TRANSPORT_FLOW_in_micro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);

  reverse_mode<double> func_BULK_PHASE_FLOW_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);
  reverse_mode<double> func_TRANSPORT_FLOW_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int num);

  void AMGXsolver_subroutine_co2_mehane(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx,
                                        int nnz_amgx);  // 混合模型方程求解以及变量更新
  void AMGXsolver_subroutine_kong(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx, int nnz_amgx);

  double macro_outlet_flow();       // 出口大孔流量
  double macro_outlet_Q();          // 出口大孔流量
  double micro_outlet_free_flow();  // 出口微孔流量
  double micro_outlet_ad_flow();    // 出口吸附量
  double micro_outlet_advec_Q();    // 出口微孔流量
  double micro_outlet_diff_Q();     // 出口吸附量
  double average_outlet_concentration();
  std::pair<double, double> area_main_Q();
  std::pair<double, double> area_side_Q();
  double Peclet_number();

  double macro_mass_loss();
  double micro_free_mass_loss();
  double micro_ad_mass_loss();

  void output_co2_methane(int n);

 public:
  ~PNMsolver()  // 析构函数，释放动态存储
  {
    delete[] dX, B;
    delete[] ia, ja, a;
    delete[] Pb, Tb_in, Tb;
    delete[] COO_A;
  }
};

void PNMsolver::output_co2_methane(int n) {
  ostringstream name;
  name << "CO2_mehante_" + Gfilename + "_" << int(n + 1) << ".vtk";
  ofstream outfile(name.str().c_str());
  outfile << "# vtk DataFile Version 2.0" << endl;
  outfile << "output.vtk" << endl;
  outfile << "ASCII " << endl;
  outfile << "DATASET POLYDATA " << endl;
  outfile << "POINTS " << pn << " float" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
  }
  // 输出孔喉连接信息
  outfile << "LINES" << "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
  for (int i = 0; i < Pb[pn - 1].full_accum; i++) {
    outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
  }
  // 输出孔体信息
  outfile << "POINT_DATA " << "\t" << pn << endl;
  outfile << "SCALARS size_pb double 1" << endl;
  outfile << "LOOKUP_TABLE table1" << endl;
  for (int i = 0; i < pn; i++) {
    if (i < macro_n) {
      outfile << Pb[i].Radiu * 2 << "\t";
    } else {
      outfile << Pb[i].Radiu << "\t";
    }
  }
  outfile << endl;
  // 输出编号信息
  outfile << "SCALARS NUMBER double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << i << endl;
  }
  // 输出压力场信息
  outfile << "SCALARS Pressure double 1" << endl;
  outfile << "LOOKUP_TABLE table3" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].pressure - outlet_pre << endl;
  }

  // 输出孔类型信息
  outfile << "SCALARS pb_type double 1" << endl;
  outfile << "LOOKUP_TABLE table4" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].type << endl;
  }

  // 输出压力场信息
  outfile << "SCALARS C_CO2 double 1" << endl;
  outfile << "LOOKUP_TABLE table5" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].mole_frac_co2 << endl;
  }
}

/*
for primary related variables
0 for ex_co2, 1 for ex_ch4, 2 for ad_co2, 3 for ad_ch4, 4 for mass_frac_co2
*/
vector<reverse_mode<double>> PNMsolver::ex_adsorption(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int Pore_id) {
  reverse_mode<double> Adsorption_ch4{0};
  reverse_mode<double> mass_frac_ch4{0};
  reverse_mode<double> mass_frac_Adsorption_ch4{0};
  reverse_mode<double> ex_Adsorption_ch4{0};

  reverse_mode<double> Adsorption_co2{0};
  reverse_mode<double> mass_frac_co2{0};
  reverse_mode<double> mass_frac_Adsorption_co2{0};
  reverse_mode<double> ex_Adsorption_co2{0};

  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mole_mass_bulk{0};

  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Adsorption_co2 = n_max_ad_co2 * (K_langmuir_co2 * Pi * Wi) / (1 + K_langmuir_co2 * Pi * Wi + K_langmuir_ch4 * Pi * (1 - Wi));
  mass_frac_co2 = Wi * MOLE_MASS_CO2 / (MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi));

  Adsorption_ch4 = n_max_ad_ch4 * (K_langmuir_ch4 * Pi * (1 - Wi)) / (1 + K_langmuir_co2 * Pi * Wi + K_langmuir_ch4 * Pi * (1 - Wi));
  mass_frac_ch4 = (1 - mass_frac_co2);

  mass_frac_Adsorption_co2 = Adsorption_co2 / (Adsorption_co2 + Adsorption_ch4);
  mass_frac_Adsorption_ch4 = (1 - mass_frac_Adsorption_co2);
  ex_Adsorption_ch4 = Adsorption_ch4 * (1 - mass_frac_ch4 * Mass_density_bulk / (mass_frac_Adsorption_ch4 * Rho_ad));
  ex_Adsorption_co2 = Adsorption_co2 * (1 - mass_frac_co2 * Mass_density_bulk / (mass_frac_Adsorption_co2 * Rho_ad));

  vector<reverse_mode<double>> RETURN;
  RETURN.push_back(ex_Adsorption_co2);
  RETURN.push_back(ex_Adsorption_ch4);
  RETURN.push_back(Adsorption_co2);
  RETURN.push_back(Adsorption_ch4);
  RETURN.push_back(mass_frac_co2);
  return RETURN;
}

/*
for old parameters
0 for ex_co2, 1 for ex_ch4, 2 for ad_co2, 3 for ad_ch4, 4 for mass_frac_co2
*/
vector<double> PNMsolver::ex_adsorption(double Pi, double Wi, int Pore_id) {
  double Adsorption_ch4{0};
  double mass_frac_ch4{0};
  double mass_frac_Adsorption_ch4{0};
  double ex_Adsorption_ch4{0};

  double Adsorption_co2{0};
  double mass_frac_co2{0};
  double mass_frac_Adsorption_co2{0};
  double ex_Adsorption_co2{0};

  double Mass_density_bulk{0};
  double Mole_mass_bulk{0};

  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Adsorption_co2 = n_max_ad_co2 * (K_langmuir_co2 * Pi * Wi) / (1 + K_langmuir_co2 * Pi * Wi + K_langmuir_ch4 * Pi * (1 - Wi));
  mass_frac_co2 = Wi * MOLE_MASS_CO2 / (MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi));

  Adsorption_ch4 = n_max_ad_ch4 * (K_langmuir_ch4 * Pi * (1 - Wi)) / (1 + K_langmuir_co2 * Pi * Wi + K_langmuir_ch4 * Pi * (1 - Wi));
  mass_frac_ch4 = (1 - mass_frac_co2);

  mass_frac_Adsorption_co2 = Adsorption_co2 / (Adsorption_co2 + Adsorption_ch4);
  mass_frac_Adsorption_ch4 = (1 - mass_frac_Adsorption_co2);
  ex_Adsorption_ch4 = Adsorption_ch4 * (1 - mass_frac_ch4 * Mass_density_bulk / (mass_frac_Adsorption_ch4 * Rho_ad));
  ex_Adsorption_co2 = Adsorption_co2 * (1 - mass_frac_co2 * Mass_density_bulk / (mass_frac_Adsorption_co2 * Rho_ad));

  vector<double> RETURN;
  RETURN.push_back(ex_Adsorption_co2);
  RETURN.push_back(ex_Adsorption_ch4);
  RETURN.push_back(Adsorption_co2);
  RETURN.push_back(Adsorption_ch4);
  RETURN.push_back(mass_frac_co2);
  return RETURN;
}

/*
for primary related variables
*/
vector<reverse_mode<double>> PNMsolver::ex_adsorption_pro(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int Pore_id) {
  reverse_mode<double> Adsorption_ch4{0};
  reverse_mode<double> ex_Adsorption_ch4{0};

  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mole_mass_bulk{0.016};

  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
  Adsorption_ch4 = (K_langmuir_ch4 * Pi) / (1 + K_langmuir_ch4 * Pi);

  ex_Adsorption_ch4 = n_max_ad * Adsorption_ch4 * (1 - Mass_density_bulk / (Rho_ad));

  vector<reverse_mode<double>> RETURN;
  RETURN.push_back(ex_Adsorption_ch4);
  RETURN.push_back(Adsorption_ch4);
  return RETURN;
}

/*
for old parameters
*/
vector<double> PNMsolver::ex_adsorption_pro(double Pi, double Wi, int Pore_id) {
  double Adsorption_ch4{0};
  double ex_Adsorption_ch4{0};

  double Mass_density_bulk{0};
  double Mole_mass_bulk{0.016};

  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
  Adsorption_ch4 = (K_langmuir_ch4 * Pi) / (1 + K_langmuir_ch4 * Pi);

  ex_Adsorption_ch4 = n_max_ad * Adsorption_ch4 * (1 - Mass_density_bulk / (Rho_ad));

  vector<double> RETURN;
  RETURN.push_back(ex_Adsorption_ch4);
  RETURN.push_back(Adsorption_ch4);
  return RETURN;
}

reverse_mode<double> PNMsolver::conductivity_sur_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
  // 计算克努森数
  double Knusen_number{0};
  int i = throat_id;

  double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
  double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
  } else {
    Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
  }
  // 计算滑移项
  double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
  double beta = 4;
  double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
  Tb_in[i].Knusen = Knusen_number;
  Tb_in[i].Slip = Slip;
  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
    Tb_in[i].surface_diff_co2 = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    Tb_in[i].surface_diff_methane = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    Tb_in[i].Surface_diff_conduc = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
    return RETURN;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
    temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
    temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].surface_diff_co2 = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].surface_diff_methane = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
    return RETURN;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }
    temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

    temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
    temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].surface_diff_co2 = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].surface_diff_methane = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
    return RETURN;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Tb_in[i].Conductivity = Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
    Tb_in[i].surface_diff_co2 = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
    Tb_in[i].surface_diff_methane = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
    Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
    reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
    return RETURN;
  } else {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
    temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
    temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
    temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);
    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].surface_diff_co2 = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].surface_diff_methane = temp11 * temp22 / (temp11 + temp22);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
    return RETURN;
  }
}

reverse_mode<double> PNMsolver::conductivity_bulk_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
  // 计算克努森数
  double Knusen_number{0};
  int i = throat_id;

  double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
  double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
  } else {
    Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
  }
  // 计算滑移项
  double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
  double beta = 4;
  double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
  Tb_in[i].Knusen = Knusen_number;
  Tb_in[i].Slip = Slip;
  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
    Tb_in[i].Surface_diff_conduc = 0;
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
    temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
    temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    // Tb_in[i].Surface_diff_conduc = 0;
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }
    temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

    temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
    temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    // Tb_in[i].Surface_diff_conduc = 0;
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Tb_in[i].Conductivity = Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
    Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
    temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
    temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
    temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);
    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    // cout << temp1 << "\t" << temp2 <<"\t"<< Tb_in[i].Conductivity << endl;
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  }
}

reverse_mode<double> PNMsolver::conductivity_bulk(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
                                                                                                          // 计算克努森数
  double temp_sur_co2_1{0}, temp_sur_co2_2{0};
  double temp_sur_methane_1{0}, temp_sur_methane_2{0};

  double Knusen_number1{0};
  double Knusen_number2{0};
  double alpha{0};
  double beta = 4;
  double Slip1{0};
  double Slip2{0};
  int i = throat_id;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    Knusen_number1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
    Knusen_number2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
    Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
    alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
    Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
    Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
    Tb_in[i].Slip = (Slip1 + Slip2) / 2;

    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].Conductivity = Tb_in[i].Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    Knusen_number1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
    Knusen_number2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
    alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
    Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
    Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
    Tb_in[i].Slip = (Slip1 + Slip2) / 2;

    temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp2 = abs(Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
    Tb_in[i].Conductivity = Tb_in[i].Slip * temp1 * temp2 / (temp1 + temp2);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    Knusen_number1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Knusen_number2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
    Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
    alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
    Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
    Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
    Tb_in[i].Slip = (Slip1 + Slip2) / 2;

    temp2 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }
    temp1 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

    Tb_in[i].Conductivity = Tb_in[i].Slip * temp1 * temp2 / (temp1 + temp2);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Knusen_number1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Knusen_number2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
    alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
    Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
    Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
    Tb_in[i].Slip = (Slip1 + Slip2) / 2;

    Tb_in[i].Conductivity = Tb_in[i].Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else {
    Knusen_number1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Knusen_number2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
    alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
    Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
    Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
    Tb_in[i].Slip = (Slip1 + Slip2) / 2;

    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp1 = abs(Tb_in[i].Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
    temp2 = abs(Tb_in[i].Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));

    Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  }
}

reverse_mode<double> PNMsolver::conductivity_co2_DISPERSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
                                                                                                          // 计算克努森数
  double temp_sur_co2_1{0}, temp_sur_co2_2{0};
  double temp_sur_methane_1{0}, temp_sur_methane_2{0};

  int i = throat_id;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].dispersion_coe_co2 = D_dispersion * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp11 = abs(pi * pow(Pb[Tb_in[i].ID_1].Radiu, 1) * D_dispersion);
    temp22 = abs(Tb_in[i].Radiu * D_dispersion * angle2 / length2);
    Tb_in[i].dispersion_coe_co2 = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }

    temp11 = abs(Tb_in[i].Radiu * D_dispersion * angle1 / length1);
    temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * D_dispersion);
    Tb_in[i].dispersion_coe_co2 = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Tb_in[i].dispersion_coe_co2 = Tb_in[i].Radiu * D_dispersion / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  } else {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp11 = abs(Tb_in[i].Radiu * D_dispersion * angle1 / length1);
    temp22 = abs(Tb_in[i].Radiu * D_dispersion * angle2 / length2);

    Tb_in[i].dispersion_coe_co2 = temp11 * temp22 / (temp11 + temp22);
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  }
}

reverse_mode<double> PNMsolver::conductivity_mehane_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id,
                                                                     int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
                                                                                                          // 计算克努森数
  double temp_sur_co2_1{0}, temp_sur_co2_2{0};
  double temp_sur_methane_1{0}, temp_sur_methane_2{0};

  int i = throat_id;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {

    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].surface_diff_methane = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
    return conductivity;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp_sur_methane_1 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);
    temp_sur_methane_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);

    Tb_in[i].surface_diff_methane = temp_sur_methane_1 * temp_sur_methane_2 / (temp_sur_methane_1 + temp_sur_methane_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
    return conductivity;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }
    temp_sur_methane_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
    temp_sur_methane_2 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);

    Tb_in[i].surface_diff_methane = temp_sur_methane_1 * temp_sur_methane_2 / (temp_sur_methane_1 + temp_sur_methane_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
    return conductivity;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Tb_in[i].surface_diff_methane = Tb_in[i].Radiu * Ds / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
    return conductivity;
  } else {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp_sur_methane_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
    temp_sur_methane_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);

    Tb_in[i].surface_diff_methane = temp_sur_methane_1 * temp_sur_methane_2 / (temp_sur_methane_1 + temp_sur_methane_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
    return conductivity;
  }
}

reverse_mode<double> PNMsolver::conductivity_co2_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id,
                                                                  int throat_id) {
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
                                                                                                          // 计算克努森数
  double temp_sur_co2_1{0}, temp_sur_co2_2{0};
  double temp_sur_methane_1{0}, temp_sur_methane_2{0};

  int i = throat_id;
  double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

  if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
    if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
    {
      Tb_in[i].Length = 0.5 * voxel_size;
    }
    Tb_in[i].surface_diff_co2 = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
    return conductivity;
  } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp_sur_co2_1 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);
    temp_sur_co2_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);
    Tb_in[i].surface_diff_co2 = temp_sur_co2_1 * temp_sur_co2_2 / (temp_sur_co2_1 + temp_sur_co2_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
    return conductivity;
  } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
    }
    temp_sur_co2_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
    temp_sur_co2_2 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds);
    Tb_in[i].surface_diff_co2 = temp_sur_co2_1 * temp_sur_co2_2 / (temp_sur_co2_1 + temp_sur_co2_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
    return conductivity;
  } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
    Tb_in[i].surface_diff_co2 = Tb_in[i].Radiu * Ds / Tb_in[i].Length;
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
    return conductivity;
  } else {
    length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
    length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
    if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
      angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
    } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
      angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
    } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
      angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
    }
    temp_sur_co2_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
    temp_sur_co2_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);
    Tb_in[i].surface_diff_co2 = temp_sur_co2_1 * temp_sur_co2_2 / (temp_sur_co2_1 + temp_sur_co2_2);
    reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
    return conductivity;
  }
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_micro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};

  reverse_mode<double> Conductivity{0};
  reverse_mode<double> conductivity_co2_SUR{0};
  reverse_mode<double> conductivity_ch4_SUR{0};

  /* 时间项 */
  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mole_mass_bulk_old = MOLE_MASS_CO2 * Pb[Pore_id].mole_frac_co2_old + MOLE_MASS_CH4 * (1 - Pb[Pore_id].mole_frac_co2_old);
  Mass_density_bulk_old = (Pb[Pore_id].pressure_old + refer_pressure) * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);
  Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk - Mass_density_bulk_old) / dt;

  Return += Pb[Pore_id].volume *
            (ex_adsorption(Pi, Wi, Pore_id)[0] - ex_adsorption(Pb[Pore_id].pressure_old + refer_pressure, Pb[Pore_id].mole_frac_co2_old, Pore_id)[0] + ex_adsorption(Pi, Wi, Pore_id)[1] -
             ex_adsorption(Pb[Pore_id].pressure_old + refer_pressure, Pb[Pore_id].mole_frac_co2_old, Pore_id)[1]) /
            dt;

  /* 流量项 */
  size_t iCounter{0};
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
      Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Tb_in[j].ID_1].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_co2_SUR = conductivity_co2_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_ch4_SUR = conductivity_mehane_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wi, Pore_id)[2]);
      Return += conductivity_ch4_SUR * (ex_adsorption(Pi, Wi, Pore_id)[3] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Pore_id)[3]);
    } else {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
      Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_co2_SUR = conductivity_co2_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_ch4_SUR = conductivity_mehane_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wi, Pore_id)[2]);
      Return += conductivity_ch4_SUR * (ex_adsorption(Pi, Wi, Pore_id)[3] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Pore_id)[3]);
    }
  }

  return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_macro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};
  reverse_mode<double> Conductivity{0};
  size_t iCounter{0};

  /* 时间项 */
  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mole_mass_bulk_old = MOLE_MASS_CO2 * Pb[Pore_id].mole_frac_co2_old + MOLE_MASS_CH4 * (1 - Pb[Pore_id].mole_frac_co2_old);
  Mass_density_bulk_old = Pi * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);
  Return += Pb[Pore_id].volume * (Mass_density_bulk - Mass_density_bulk_old) / dt;

  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
      Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
    } else {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
      Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[ID2].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
    }
  }

  return Return;
}

reverse_mode<double> PNMsolver::func_TRANSPORT_FLOW_in_macro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};
  reverse_mode<double> Mass_frac_co2{0};
  reverse_mode<double> Mass_frac_co2_old{0};
  reverse_mode<double> Conductivity{0};
  reverse_mode<double> Conductivity_dis_co2{0};
  size_t iCounter{0};

  /* 时间项 */
  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mole_mass_bulk_old = MOLE_MASS_CO2 * Pb[Pore_id].mole_frac_co2_old + MOLE_MASS_CH4 * (1 - Pb[Pore_id].mole_frac_co2_old);
  Mass_density_bulk_old = Pi * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);

  Mass_frac_co2 = ex_adsorption(Pi, Wi, Pore_id)[4];
  Mass_frac_co2_old = ex_adsorption((Pb[Pore_id].pressure_old + refer_pressure), Pb[Pore_id].mole_frac_co2_old, Pore_id)[4];
  Return += Pb[Pore_id].volume * (Mass_density_bulk * Mass_frac_co2 - Mass_density_bulk_old * Mass_frac_co2_old) / dt;

  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
      Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);  // co2 advection term
      Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter],
                                                                                                              ID2)[4]);  // co2 dispersion term
    } else {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
      Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
      Mass_frac_co2 = ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4];
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter],
                                                                                                              ID2)[4]);  // co2 dispersion term
    }
  }
  return Return;
}

reverse_mode<double> PNMsolver::func_TRANSPORT_FLOW_in_micro(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};
  reverse_mode<double> Mass_frac_co2{0};
  reverse_mode<double> Mass_frac_co2_old{0};
  reverse_mode<double> Conductivity{0};
  reverse_mode<double> Conductivity_dis_co2{0};
  reverse_mode<double> conductivity_co2_SUR{0};

  size_t iCounter{0};

  /* 时间项 */
  Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mole_mass_bulk_old = MOLE_MASS_CO2 * Pb[Pore_id].mole_frac_co2_old + MOLE_MASS_CH4 * (1 - Pb[Pore_id].mole_frac_co2_old);
  Mass_density_bulk_old = Pi * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);

  Mass_frac_co2 = ex_adsorption(Pi, Wi, Pore_id)[4];
  Mass_frac_co2_old = ex_adsorption((Pb[Pore_id].pressure_old + refer_pressure), Pb[Pore_id].mole_frac_co2_old, Pore_id)[4];
  Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk * Mass_frac_co2 - Mass_density_bulk_old * Mass_frac_co2_old) / dt;
  Return += Pb[Pore_id].volume * (ex_adsorption(Pi, Wi, Pore_id)[0] - ex_adsorption((Pb[Pore_id].pressure_old + refer_pressure), Pb[Pore_id].mole_frac_co2_old, Pore_id)[0]) / dt;

  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
      Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);  // co2 advection term
      Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter],
                                                                                                              ID2)[4]);  // co2 dispersion term
      Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[2]);
    } else {
      Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
      Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
      Mass_frac_co2 = ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4];
      Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter],
                                                                                                              ID2)[4]);  // co2 dispersion term
      Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[2]);
    }
  }
  return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_micro_produc(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};

  reverse_mode<double> Conductivity{0};
  reverse_mode<double> conductivity_ch4_SUR{0};

  /* 时间项 */
  Mole_mass_bulk = 0.016;
  Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mole_mass_bulk_old = 0.016;
  Mass_density_bulk_old = (Pb[Pore_id].pressure_old + refer_pressure) * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);
  Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk - Mass_density_bulk_old) / dt;

  Return += Pb[Pore_id].volume * (ex_adsorption_pro(Pi, Wi, Pore_id)[0] - ex_adsorption_pro((Pb[Pore_id].pressure_old + refer_pressure), Pb[Pore_id].mole_frac_co2_old, Pore_id)[0]) / dt;

  /* 流量项 */
  size_t iCounter{0};
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Tb_in[j].ID_1].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_ch4_SUR = conductivity_sur_test(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += conductivity_ch4_SUR * (ex_adsorption_pro(Pi, Wi, Pore_id)[1] - ex_adsorption_pro(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[1]);
    } else {
      Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
      conductivity_ch4_SUR = conductivity_sur_test(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
      Return += conductivity_ch4_SUR * (ex_adsorption_pro(Pi, Wi, Pore_id)[1] - ex_adsorption_pro(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[1]);
    }
  }
  return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_macro_produc(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Mass_density_bulk{0};
  reverse_mode<double> Mass_density_bulk_old{0};
  reverse_mode<double> Mole_mass_bulk{0};
  reverse_mode<double> Mole_mass_bulk_old{0};
  reverse_mode<double> Conductivity{0};
  size_t iCounter{0};

  /* 时间项 */
  Mass_density_bulk = Pi * 0.016 / (Pb[Pore_id].compre * 8.314 * Temperature);

  Mass_density_bulk_old = (Pb[Pore_id].pressure_old + refer_pressure) * 0.016 / (Pb[Pore_id].compre_old * 8.314 * Temperature);
  Return += Pb[Pore_id].volume * (Mass_density_bulk - Mass_density_bulk_old) / dt;

  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Mass_density_bulk = Pi * 0.016 / (Pb[Pore_id].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
    } else {
      Mass_density_bulk = Pjs[iCounter] * 0.016 / (Pb[ID2].compre * 8.314 * Temperature);
      Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
    }
  }
  return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> RETURN;
  // size_t counter{0};
  size_t iCounter{0};
  /* 时间项 */
  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Tb[j].ID_2 < inlet)  // 大孔进口
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      } else {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      }
    } else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)  // 大孔出口
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      } else {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      }
    } else if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)  // 微孔进口边界
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      } else {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      }
    } else if (Tb[j].ID_2 >= pn - m_outlet)  // 微孔出口边界
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      } else {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      }
    } else {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      } else {
        RETURN += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j) * (Pi - Pjs[iCounter]);
      }
    }
  }
  return RETURN;
}

reverse_mode<double> PNMsolver::conductivity_bulk_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {
  int i = throat_id;

  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp3 = 0,
         temp4 = 0;  // 两点流量计算中的临时存储变量
  double length1{0}, length2{0};
  // 计算努森数 滑移项
  double Knusen_number_ID1{0};
  double Knusen_number_ID2{0};
  double Slip_ID1{0};
  double Slip_ID2{0};
  double Rho_ID1{0};
  double Rho_ID2{0};
  double K_ID1{0};
  double K_ID2{0};
  double Apparent_K_ID1{0};
  double Apparent_K_ID2{0};

  Rho_ID1 = 0.016 * (Pb[Tb_in[i].ID_1].pressure + refer_pressure) / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
  Rho_ID2 = 0.016 * (Pb[Tb_in[i].ID_2].pressure + refer_pressure) / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
  if (Pb[Tb_in[i].ID_1].type == 0)  // clay_HP
  {
    Apparent_K_ID1 = Pb[Tb_in[i].ID_1].km;
  } else if (Pb[Tb_in[i].ID_1].type == 1)  // clay_LP
  {
    Apparent_K_ID1 = Pb[Tb_in[i].ID_1].km;
  } else if (Pb[Tb_in[i].ID_1].type == 2)  // macro pores
  {
    // Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure
    // + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 *
    // Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
    Apparent_K_ID1 = pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 8;
  } else if (Pb[Tb_in[i].ID_1].type == 3)  // crack
  {
    Apparent_K_ID1 = pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 12;
  } else if (Pb[Tb_in[i].ID_1].type == 4)  // OM_type_HP
  {
    Apparent_K_ID1 = Pb[Tb_in[i].ID_1].km;
  } else if (Pb[Tb_in[i].ID_1].type == 5)  // OM_type_LP
  {
    Apparent_K_ID1 = Pb[Tb_in[i].ID_1].km;
  }

  // 6.96e-9
  if (Pb[Tb_in[i].ID_2].type == 0)  // clay
  {
    Apparent_K_ID2 = Pb[Tb_in[i].ID_2].km;
  } else if (Pb[Tb_in[i].ID_2].type == 1)  // clay_LP
  {
    Apparent_K_ID2 = Pb[Tb_in[i].ID_2].km;
  } else if (Pb[Tb_in[i].ID_2].type == 2)  // macro pores
  {
    Apparent_K_ID2 = pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 8;
  } else if (Pb[Tb_in[i].ID_2].type == 3)  // crack
  {
    Apparent_K_ID2 = pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 12;
  } else if (Pb[Tb_in[i].ID_2].type == 4)  // OM_type1
  {
    Apparent_K_ID2 = Pb[Tb_in[i].ID_2].km;
  } else if (Pb[Tb_in[i].ID_2].type == 5)  // OM_type2
  {
    Apparent_K_ID2 = Pb[Tb_in[i].ID_2].km;
  }
  /*debug*/
  // Pb[Tb_in[i].ID_1].visco = gas_vis;
  // Pb[Tb_in[i].ID_2].visco = gas_vis;
  if (Pb[Tb_in[i].ID_1].type == 2 && Pb[Tb_in[i].ID_2].type == 2) {
    temp1 = Apparent_K_ID1 * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / (Pb[Tb_in[i].ID_1].visco * Pb[Tb_in[i].ID_1].Radiu);
    temp2 = pi * pow(Tb_in[i].Radiu, 4) / (8 * (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2 * Tb_in[i].Length);
    temp3 = Apparent_K_ID2 * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / (Pb[Tb_in[i].ID_2].visco * Pb[Tb_in[i].ID_2].Radiu);
    Tb_in[i].Conductivity = temp1 * temp2 * temp3 / (temp1 * temp2 + temp2 * temp3 + temp1 * temp3);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
  } else {
    length1 = Tb_in[i].Length * Pb[Tb_in[i].ID_1].Radiu / (Pb[Tb_in[i].ID_1].Radiu + Pb[Tb_in[i].ID_2].Radiu);
    length2 = Tb_in[i].Length * Pb[Tb_in[i].ID_2].Radiu / (Pb[Tb_in[i].ID_1].Radiu + Pb[Tb_in[i].ID_2].Radiu);

    temp1 = Apparent_K_ID1 * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / (Pb[Tb_in[i].ID_1].visco * Pb[Tb_in[i].ID_1].Radiu);
    temp2 = Apparent_K_ID1 * pi * pow(Tb_in[i].Radiu, 2) / (Pb[Tb_in[i].ID_1].visco * length1);
    temp3 = Apparent_K_ID2 * pi * pow(Tb_in[i].Radiu, 2) / (Pb[Tb_in[i].ID_2].visco * length2);
    temp4 = Apparent_K_ID2 * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / (Pb[Tb_in[i].ID_2].visco * Pb[Tb_in[i].ID_2].Radiu);

    Tb_in[i].Conductivity = temp1 * temp2 * temp3 * temp4 / (temp2 * temp3 * temp4 + temp1 * temp3 * temp4 + temp1 * temp2 * temp4 + temp1 * temp2 * temp3);
    reverse_mode<double> conductivity = Tb_in[i].Conductivity;
    return conductivity;
    /*rong_debug*/
    // Tb_in[i].Conductivity = 1e-30;
  }
}

reverse_mode<double> PNMsolver::conductivity_co2_DISPERSION_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id, int throat_id) {

  int i = throat_id;
  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp3 = 0,
         temp4 = 0;  // 两点流量计算中的临时存储变量
  double length1{0}, length2{0};
  // 计算努森数 滑移项
  double Knusen_number_ID1{0};
  double Knusen_number_ID2{0};
  double Slip_ID1{0};
  double Slip_ID2{0};
  double Rho_ID1{0};
  double Rho_ID2{0};
  double K_ID1{0};
  double K_ID2{0};
  double Apparent_K_ID1{0};
  double Apparent_K_ID2{0};

  Rho_ID1 = 0.016 * (Pb[Tb_in[i].ID_1].pressure + refer_pressure) / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
  Rho_ID2 = 0.016 * (Pb[Tb_in[i].ID_2].pressure + refer_pressure) / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
  if (Pb[Tb_in[i].ID_1].type == 0)  // clay_HP
  {
    Apparent_K_ID1 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_1].type == 1)  // clay_LP
  {
    Apparent_K_ID1 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_1].type == 2)  // macro pores
  {
    Apparent_K_ID1 = kong::D_dispersion_macro;
  } else if (Pb[Tb_in[i].ID_1].type == 3)  // crack
  {
    Apparent_K_ID1 = kong::D_dispersion_macro;
  } else if (Pb[Tb_in[i].ID_1].type == 4)  // OM_type_HP
  {
    Apparent_K_ID1 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_1].type == 5)  // OM_type_LP
  {
    Apparent_K_ID1 = kong::D_dispersion_micro;
  }

  // 6.96e-9
  if (Pb[Tb_in[i].ID_2].type == 0)  // clay
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_2].type == 1)  // clay_LP
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_2].type == 2)  // macro pores
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_2].type == 3)  // crack
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_2].type == 4)  // OM_type1
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  } else if (Pb[Tb_in[i].ID_2].type == 5)  // OM_type2
  {
    Apparent_K_ID2 = kong::D_dispersion_micro;
  }
  /*debug*/
  // Pb[Tb_in[i].ID_1].visco = gas_vis;
  // Pb[Tb_in[i].ID_2].visco = gas_vis;
  if (Pb[Tb_in[i].ID_1].type == 2 && Pb[Tb_in[i].ID_2].type == 2) {
    temp1 = Apparent_K_ID1 * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / (Pb[Tb_in[i].ID_1].Radiu);
    temp2 = kong::D_dispersion_macro * pi * pow(Tb_in[i].Radiu, 2) / (Tb_in[i].Length);
    temp3 = Apparent_K_ID2 * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / (Pb[Tb_in[i].ID_2].Radiu);
    Tb_in[i].dispersion_coe_co2 = temp1 * temp2 * temp3 / (temp1 * temp2 + temp2 * temp3 + temp1 * temp3);
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
  } else {
    length1 = Tb_in[i].Length * Pb[Tb_in[i].ID_1].Radiu / (Pb[Tb_in[i].ID_1].Radiu + Pb[Tb_in[i].ID_2].Radiu);
    length2 = Tb_in[i].Length * Pb[Tb_in[i].ID_2].Radiu / (Pb[Tb_in[i].ID_1].Radiu + Pb[Tb_in[i].ID_2].Radiu);

    temp1 = Apparent_K_ID1 * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / (Pb[Tb_in[i].ID_1].Radiu);
    temp2 = Apparent_K_ID1 * pi * pow(Tb_in[i].Radiu, 2) / (length1);
    temp3 = Apparent_K_ID2 * pi * pow(Tb_in[i].Radiu, 2) / (length2);
    temp4 = Apparent_K_ID2 * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / (Pb[Tb_in[i].ID_2].Radiu);

    Tb_in[i].dispersion_coe_co2 = temp1 * temp2 * temp3 * temp4 / (temp2 * temp3 * temp4 + temp1 * temp3 * temp4 + temp1 * temp2 * temp4 + temp1 * temp2 * temp3);
    reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
    return conductivity;
    /*rong_debug*/
    // Tb_in[i].Conductivity = 1e-30;
  }
}

reverse_mode<double> PNMsolver::func_TRANSPORT_FLOW_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int Pore_id) {
  reverse_mode<double> Return{0};
  reverse_mode<double> Conductivity{0};
  reverse_mode<double> Conductivity_dis_co2{0};
  size_t iCounter{0};

  /* 时间项 */
  Return += Pb[Pore_id].volume * (Wi - Pb[Pore_id].mole_frac_co2_old) / dt;

  /* 流量项 */
  int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++) {
    // 判断是否连接同一微孔区域 不用判断是否是进出口
    // 不关于进出口的变量求导就行了
    if (Tb_in[j].ID_2 != ID2) {
      iCounter++;
      ID2 = Tb_in[j].ID_2;
    }

    if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure) {
      Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION_kong(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Wi * Conductivity * (Pi - Pjs[iCounter]);  // co2 advection term
      Return += Conductivity_dis_co2 * (Wi - Wjs[iCounter]);
    } else {
      Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, Pore_id, j);
      Conductivity_dis_co2 = conductivity_co2_DISPERSION_kong(Pi, Pjs, Wi, Wjs, Pore_id, j);

      Return += Wjs[iCounter] * Conductivity * (Pi - Pjs[iCounter]);  // co2 advection term
      Return += Conductivity_dis_co2 * (Wi - Wjs[iCounter]);
    }
  }
  return Return;
}

void PNMsolver::para_cal_co2_methane()  // 喉道长度等相关参数计算
{
  // 计算孔隙的体积
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    if (Pb[i].type == 0) {
      Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;  // 孔隙网络单元
    } else if (Pb[i].type == 1) {
      Pb[i].volume = pow(Pb[i].Radiu, 3);  // 正方形微孔单元
    } else {
      Pb[i].volume = pow(Pb[i].Radiu, 3) / 2;  // 2×2×1、1×2×2和2×1×2的微孔网格
    }
  }

  // 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    // Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 *
    // pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
    Pb[i].compre = 1;
    Pb[i].compre_old = Pb[i].compre;
    Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
    Pb[i].visco_old = Pb[i].visco;
  }

  // Total gas content
  double compre_1 = compre(inlet_pre + refer_pressure);
  double compre_2 = compre(outlet_pre);
  double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
  double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_macro)
#endif
  for (int i = inlet; i < macro_n - outlet; i++) {
    total_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    total_micro_free += (inlet_pre + refer_pressure) * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) -
                        outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
    total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
  }

  total_p = total_macro + total_micro_free + total_micro_ad;
  cout << "total_macro = " << total_macro << endl;
  cout << "total_micro_free = " << total_micro_free << endl;
  cout << "total_micro_ad = " << total_micro_ad << endl;
  cout << "total_p = " << total_p << endl;

  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
    }
  }

  // full_coord
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }

  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  Pb[0].full_accum = Pb[0].full_coord;

  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }

#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("para_cal diff = %.16g\n", end - start);
#endif

  coolist.resize(op + mp);   // 非进出口全配位数
  coolist3.resize(op + mp);  // 非进出口的局部指标
  coolist4.resize(op + mp);  // 非进出口的全局指标
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    int counter{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      int counter{0};
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }

  coolist2.resize(op + mp);  // 非进出口累计全配位数
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < coolist2.size(); i++) {
    std::vector<int>::iterator it = coolist.begin() + i;
    coolist2[i] = accumulate(coolist.begin(), it, 0);
  }

  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp) * 2];
  B = new double[(op + mp) * 2];

  ia = new int[(op + mp) * 2 + 1];
  ja = new int[NA * 3];

  a = new double[NA * 3];

  COO_A = new Acoo[NA * 3];
}

void PNMsolver::CO2_methane_matrix() {
  /* -------------------------------------------------------------------------------------
   */
  /* BULK PHASE EQUATION SOLEVR */
  /* -------------------------------------------------------------------------------------
   */

  /* -------------------------------------------------------------------------------------
   */
  /* 大孔组装 */
  /* -------------------------------------------------------------------------------------
   */
  int counter = 0;
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < NA * 3; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp) * 2; i++) {
    B[i] = 0;
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_BULK_PHASE_FLOW_in_macro(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);
    B[i - inlet] = -F.val();
    COO_A[i - inlet].row = i - inlet;
    COO_A[i - inlet].col = i - inlet;
    COO_A[i - inlet].val = Pi.d(0);

    size_t counter{0};   // 跳过进出口
    size_t counter1{0};  // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))  // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))  // 连接的是微孔
      {
        COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_BULK_PHASE_FLOW_in_micro(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro] = -F.val();
    COO_A[i - para_macro].row = i - para_macro;
    COO_A[i - para_macro].col = i - para_macro;
    COO_A[i - para_macro].val = Pi.d(0);

    size_t counter{0};   // 跳过进出口
    size_t counter1{0};  // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))  // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))  // 连接的是微孔
      {
        COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

  /* -------------------------------------------------------------------------------------
   */
  /* TRANSPORT EQUATION SOLEVR */
  /* -------------------------------------------------------------------------------------
   */

  /* -------------------------------------------------------------------------------------
   */
  /* 大孔组装 */
  /* -------------------------------------------------------------------------------------
   */
  counter = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];
    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_TRANSPORT_FLOW_in_macro(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - inlet + op + mp] = -F.val();
    COO_A[i - inlet + NA].row = i - inlet + op + mp;
    COO_A[i - inlet + NA].col = i - inlet;
    COO_A[i - inlet + NA].val = Pi.d(0);

    COO_A[i - inlet + 2 * NA].row = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].col = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].val = Wi.d(0);

    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_TRANSPORT_FLOW_in_micro(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro + op + mp] = -F.val();
    COO_A[i - para_macro + 1 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 1 * NA].col = i - para_macro;
    COO_A[i - para_macro + 1 * NA].val = Pi.d(0);

    COO_A[i - para_macro + 2 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].col = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].val = Wi.d(0);
    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("matrix diff = %.16g\n", end - start);
#endif
  // ofstream COOA_OUT("COOA_ad_unsorted.txt");

  // for (size_t i = 0; i < 4 * NA - coolist2[op]; i++)
  // {
  // 	COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val
  // << endl;
  // }

  // ofstream B_OUT("B_OUT.txt");

  // for (size_t i = 0; i < 2 * (op + mp); i++)
  // {
  // 	B_OUT << B[i] << endl;
  // }
}

void PNMsolver::kong_matrix() {
  /* -------------------------------------------------------------------------------------
   */
  /* BULK PHASE EQUATION SOLEVR */
  /* -------------------------------------------------------------------------------------
   */

  /* -------------------------------------------------------------------------------------
   */
  /* 大孔组装 */
  /* -------------------------------------------------------------------------------------
   */
  int counter = 0;
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < NA * 3; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp) * 2; i++) {
    B[i] = 0;
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);
    B[i - inlet] = -F.val();
    COO_A[i - inlet].row = i - inlet;
    COO_A[i - inlet].col = i - inlet;
    COO_A[i - inlet].val = Pi.d(0);

    size_t counter{0};   // 跳过进出口
    size_t counter1{0};  // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))  // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))  // 连接的是微孔
      {
        COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro] = -F.val();
    COO_A[i - para_macro].row = i - para_macro;
    COO_A[i - para_macro].col = i - para_macro;
    COO_A[i - para_macro].val = Pi.d(0);

    size_t counter{0};   // 跳过进出口
    size_t counter1{0};  // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))  // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))  // 连接的是微孔
      {
        COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

  /* -------------------------------------------------------------------------------------
   */
  /* TRANSPORT EQUATION SOLEVR */
  /* -------------------------------------------------------------------------------------
   */

  /* -------------------------------------------------------------------------------------
   */
  /* 大孔组装 */
  /* -------------------------------------------------------------------------------------
   */
  counter = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];
    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_TRANSPORT_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - inlet + op + mp] = -F.val();
    COO_A[i - inlet + NA].row = i - inlet + op + mp;
    COO_A[i - inlet + NA].col = i - inlet;
    COO_A[i - inlet + NA].val = Pi.d(0);

    COO_A[i - inlet + 2 * NA].row = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].col = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].val = Wi.d(0);

    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].mole_frac_co2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)  // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
    }

    F = func_TRANSPORT_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro + op + mp] = -F.val();
    COO_A[i - para_macro + 1 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 1 * NA].col = i - para_macro;
    COO_A[i - para_macro + 1 * NA].val = Pi.d(0);

    COO_A[i - para_macro + 2 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].col = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].val = Wi.d(0);
    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = Pjs[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Wjs[counter].d(0);
        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("matrix diff = %.16g\n", end - start);
#endif
  // ofstream COOA_OUT("COOA_ad_unsorted.txt");

  // for (size_t i = 0; i < 4 * NA - coolist2[op]; i++)
  // {
  // 	COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val
  // << endl;
  // }

  // ofstream B_OUT("B_OUT.txt");

  // for (size_t i = 0; i < 2 * (op + mp); i++)
  // {
  // 	B_OUT << B[i] << endl;
  // }
}

double PNMsolver::Function_Slip(double knusen) {
  double alpha_om = 1.358 * 2 / pi * atan(4 * pow(knusen, 0.4));
  double beta_om = 4;
  double Slip_om = (1 + alpha_om * knusen) * (1 + beta_om * knusen / (1 + knusen));
  return Slip_om;
}

double PNMsolver::Function_Slip_clay(double knusen) {
  double alpha_c = 1.5272 * 2 / pi * atan(2.5 * pow(knusen, 0.5));
  double beta_c = 6;
  double Slip_c = (1 + alpha_c * knusen) * (1 + beta_c * knusen / (1 + knusen));
  return Slip_c;
}

#include <string>

void PNMsolver::AMGX_solver_C_kong_PNM() {
  Flag_eigen = false;
  Flag_Hybrid = false;
  auto start1 = high_resolution_clock::now();
  double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

  int n{1};
  int inter_n{0};                         // The interation of outer loop of Newton-raphoon method
  double total_flow = 0;                  // accumulation production
  ofstream outfile(Gfilename, ios::app);  // output permeability;
  memory();
  Paramentinput();
  if (time_step != 0) {
    initial_condition(1);
    n = percentage_production_counter;  // label of output file
  } else {
    initial_condition();
  }

  para_cal_kong();
  kong_matrix();
  Matrix_COO2CSR();
  // begin AMGX initialization
  AMGX_initialize();

  AMGX_config_handle config;
  AMGX_config_create_from_file(&config, "solver.json");  // 200

  AMGX_resources_handle rsrc;
  AMGX_resources_create_simple(&rsrc, config);

  AMGX_solver_handle solver;
  AMGX_matrix_handle A_amgx;
  AMGX_vector_handle b_amgx;
  AMGX_vector_handle solution_amgx;

  AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, config);
  AMGX_matrix_create(&A_amgx, rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&b_amgx, rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&solution_amgx, rsrc, AMGX_mode_dDDI);

  int n_amgx = (op + mp) * 2;
  int nnz_amgx = ia[(op + mp) * 2];
  AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
  AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
  AMGX_pin_memory(a, nnz_amgx * sizeof(double));
  AMGX_pin_memory(B, sizeof(double) * n_amgx);
  AMGX_pin_memory(dX, sizeof(double) * n_amgx);

  auto stop3 = high_resolution_clock::now();
  auto duration3 = duration_cast<milliseconds>(stop3 - start1);
  outfile << "inner loop = " << inter_n << "\t" << "machine_time = " << duration3.count() / 1000 + machine_time << "\t" << "physical_time = " << time_all << "\t"
          << "dimensionless_time = " << time_all / (2000 * voxel_size) * ((area_main_Q().second + area_side_Q().second) / pow(1745 * voxel_size, 2) / 0.2) << "\t" << "v_main = " << area_main_Q().first
          << "\t" << "v_side = " << area_side_Q().first << "\t" << "dt = " << dt << "\t"
          << "Peclet_MAIN = " << area_main_Q().second / (pow(1745 * voxel_size, 2) * 0.2 * 3641 / 4900) / kong::D_dispersion_macro * 10e-6 << "\t"
          << "Peclet_side = " << area_side_Q().second / (pow(1745 * voxel_size, 2) * 0.2 * 1259 / 4900) / kong::D_dispersion_macro * 10e-6 << "\t"
          << "average_outlet_c = " << average_outlet_concentration() << "\t" << endl;

  output_co2_methane(time_step - 1);  // 初始状态
  // end AMGX initialization
  // ************ begin AMGX solver ************
  int nn{1};
  AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
  do {
    inter_n = 0;
    do {
      kong_matrix();
      Matrix_COO2CSR();
      AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
      inter_n++;
      cout << "Inf_norm = " << norm_inf << "\t\t" << "dt = " << dt << "\t\t" << "inner loop = " << inter_n << "\t\t" << "outer loop = " << time_step + 1 << endl;
      cout << endl;
    } while (norm_inf > eps);

    for (int i = 0; i < inlet; i++) {
      bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
      for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
        reverse_mode<double> Pi, Wi;
        reverse_mode<double>* Pjs;
        reverse_mode<double>* Wjs;
        double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
        double Aij = pi * pow(Tb[j].Radiu, 2);
        if (exists) {
          double vij = 1.29 * 0.01 / 60;                                 // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
        } else {
          double vij = 1.04 * 0.01 / 60;                                 // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
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
          double vij = 1.29 * 0.01 / 60;                                 // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
        } else {
          double vij = 1.04 * 0.01 / 60;                                 // m/s
          Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
        }
      }
    }

    time_all += dt;

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start1);
    outfile << "inner loop = " << inter_n << "\t" << "machine_time = " << duration3.count() / 1000 + machine_time << "\t" << "physical_time = " << time_all << "\t"
            << "dimensionless_time = " << time_all / (2000 * voxel_size) * ((area_main_Q().second + area_side_Q().second) / (pow(1745 * voxel_size, 2) * 0.2)) << "\t"
            << "v_main = " << area_main_Q().first << "\t" << "v_side = " << area_side_Q().first << "\t" << "dt = " << dt << "\t"
            << "Peclet_MAIN = " << area_main_Q().second / (pow(1745 * voxel_size, 2) * 0.2 * 3641 / 4900) / kong::D_dispersion_macro * 10e-6 << "\t"
            << "Peclet_side = " << area_side_Q().second / (pow(1745 * voxel_size, 2) * 0.2 * 1259 / 4900) / kong::D_dispersion_macro * 10e-6 << "\t"
            << "average_outlet_c = " << average_outlet_concentration() << "\t" << endl;

    for (int i = 0; i < pn; i++) {
      Pb[i].pressure_old = Pb[i].pressure;
      Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
    }

    if (inter_n < 10 && dt < 0.5) {
      dt = dt * 2;
    }

    output_co2_methane(time_step);

    time_step++;
  } while (average_outlet_concentration() < 0.99);
  output_co2_methane(time_step);
  outfile.close();
  auto stop1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(stop1 - start1);
  cout << "Time-consuming = " << duration1.count() << " MS" << endl;
  ofstream out("calculate time.txt");
  out << duration1.count();
  out.close();

  /***********************销毁AMGX***************************/
  AMGX_unpin_memory(ia);
  AMGX_unpin_memory(ja);
  AMGX_unpin_memory(a);
  AMGX_unpin_memory(B);
  // AMGX_unpin_memory(X);
  AMGX_unpin_memory(dX);
  // AMGX_unpin_memory(Xn);

  AMGX_solver_destroy(solver);
  AMGX_vector_destroy(b_amgx);
  AMGX_vector_destroy(solution_amgx);
  AMGX_matrix_destroy(A_amgx);
  AMGX_resources_destroy(rsrc);
  AMGX_config_destroy(config);
  AMGX_finalize();
  // ************ end AMGX solver ************
}

void PNMsolver::AMGX_solver_CO2_methane() {
  Flag_eigen = false;
  auto start1 = high_resolution_clock::now();
  double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

  int n{1};
  int inter_n{0};                                      // The interation of outer loop of Newton-raphoon method
  double total_flow = 0;                               // accumulation production
  ofstream outfile("CO2_methane_Pe_1.txt", ios::app);  // output permeability;

  Function_DS(inlet_pre + refer_pressure);
  memory();
  Paramentinput();
  if (time_step != 0) {
    initial_condition(1);
    n = percentage_production_counter;  // label of output file
  } else {
    initial_condition();
  }
  para_cal_co2_methane();
  CO2_methane_matrix();
  Matrix_COO2CSR();
  // begin AMGX initialization
  AMGX_initialize();

  AMGX_config_handle config;
  AMGX_config_create_from_file(&config,
                               "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/1.json");  // 200

  AMGX_resources_handle rsrc;
  AMGX_resources_create_simple(&rsrc, config);

  AMGX_solver_handle solver;
  AMGX_matrix_handle A_amgx;
  AMGX_vector_handle b_amgx;
  AMGX_vector_handle solution_amgx;

  AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, config);
  AMGX_matrix_create(&A_amgx, rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&b_amgx, rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&solution_amgx, rsrc, AMGX_mode_dDDI);

  int n_amgx = (op + mp) * 2;
  int nnz_amgx = ia[(op + mp) * 2];
  AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
  AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
  AMGX_pin_memory(a, nnz_amgx * sizeof(double));
  AMGX_pin_memory(B, sizeof(double) * n_amgx);
  AMGX_pin_memory(dX, sizeof(double) * n_amgx);

  auto stop3 = high_resolution_clock::now();
  auto duration3 = duration_cast<milliseconds>(stop3 - start1);
  outfile << "inner loop = " << inter_n << "\t" << "machine_time = " << duration3.count() / 1000 + machine_time << "\t" << "physical_time = " << time_all << "\t" << "dt = " << dt << "\t"
          << "Q_outlet_macro = " << Q_outlet_macro << "\t" << "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t" << "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
          << "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t" << "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"
          << "mass_conservation_error = " << 0 << "\t" << "macro_loss = " << free_macro_loss << "\t" << "free_micro_loss = " << free_micro_loss << "\t" << "ad_micro_loss = " << ad_micro_loss << "\t"
          << "acu_flow_macro = " << acu_flow_macro << "\t" << "acu_free_micro = " << acu_free_micro << "\t" << "acu_ad_micro = " << acu_ad_micro << "\t"
          << "total_flow / total_p = " << total_flow / total_p << "\t" << "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t" << "acu_flow_micro / total_p = " << acu_free_micro / total_p
          << "\t" << "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t" << endl;
  output_co2_methane(time_step - 1);  // 初始状态
  // end AMGX initialization
  // ************ begin AMGX solver ************
  int nn{1};
  AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
  do {
    inter_n = 0;
    do {
      CO2_methane_matrix();
      Matrix_COO2CSR();
      AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
      inter_n++;
      cout << "Inf_norm = " << norm_inf << "\t\t" << "dt = " << dt << "\t\t" << "inner loop = " << inter_n << "\t\t" << "outer loop = " << time_step + 1 << endl;
      cout << endl;
    } while (norm_inf > eps);

    time_all += dt;
    acu_flow_macro = 0;
    acu_free_micro = 0;
    acu_ad_micro = 0;
    for (int i = inlet; i < macro_n - outlet; i++) {
      double compre_1 = compre(inlet_pre + refer_pressure);
      acu_flow_macro +=
          (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
    }
    for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
      double compre_old = compre(inlet_pre + refer_pressure);
      double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
      double n_ad_old = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
      acu_free_micro += (porosity - n_ad_old / Rho_ad) * (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) -
                        (porosity - n_ad_new / Rho_ad) * (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
    }
    for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
      acu_ad_micro += Pb[i].volume * n_max_ad * 1000 *
                      abs(K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)) -
                          K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));  // 微孔累计产气质量 单位g
    }

    total_flow = acu_flow_macro + acu_free_micro + acu_ad_micro;

    free_macro_loss = macro_mass_loss();
    free_micro_loss = micro_free_mass_loss();
    ad_micro_loss = micro_ad_mass_loss();

    Q_outlet_macro = macro_outlet_flow();
    Q_outlet_free_micro = micro_outlet_free_flow();
    Q_outlet_ad_micro = micro_outlet_ad_flow();
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start1);
    outfile << "inner loop = " << inter_n << "\t" << "machine_time = " << duration2.count() / 1000 + machine_time << "\t" << "physical_time = " << time_all << "\t" << "dt = " << dt << "\t"
            << "Q_outlet_macro = " << Q_outlet_macro << "\t" << "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t" << "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
            << "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

            << "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

            << "mass_conservation_error = "
            << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
            << "macro_loss = " << free_macro_loss << "\t" << "free_micro_loss = " << free_micro_loss << "\t" << "ad_micro_loss = " << ad_micro_loss << "\t"

            << "acu_flow_macro = " << acu_flow_macro << "\t" << "acu_free_micro = " << acu_free_micro << "\t" << "acu_ad_micro = " << acu_ad_micro << "\t"
            << "total_flow / total_p = " << total_flow / total_p << "\t" << "acu_flow_macro / total_sub-resolution poresp = " << acu_flow_macro / total_p << "\t"
            << "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t" << "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t" << "eps = " << eps << "\t" << endl;

    for (int i = 0; i < pn; i++) {
      Pb[i].pressure_old = Pb[i].pressure;
      Pb[i].compre_old = Pb[i].compre;
      Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
    }

    // if (inter_n < 5)
    // {
    // 	dt = dt * 2;
    // }

    output_co2_methane(time_step);

    time_step++;
  } while (true);
  output_co2_methane(time_step);
  outfile.close();
  auto stop1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(stop1 - start1);
  cout << "Time-consuming = " << duration1.count() << " MS" << endl;
  ofstream out("calculate time.txt");
  out << duration1.count();
  out.close();

  /***********************销毁AMGX***************************/
  AMGX_unpin_memory(ia);
  AMGX_unpin_memory(ja);
  AMGX_unpin_memory(a);
  AMGX_unpin_memory(B);
  // AMGX_unpin_memory(X);
  AMGX_unpin_memory(dX);
  // AMGX_unpin_memory(Xn);

  AMGX_solver_destroy(solver);
  AMGX_vector_destroy(b_amgx);
  AMGX_vector_destroy(solution_amgx);
  AMGX_matrix_destroy(A_amgx);
  AMGX_resources_destroy(rsrc);
  AMGX_config_destroy(config);
  AMGX_finalize();
  // ************ end AMGX solver ************
}

void PNMsolver::CSR2COO() {
  // 将CSR格式转换为COO格式
  int k = 0;
  for (size_t i = 0; i < op + mp + 1; i++) {
    ia[i] += 1;
  }

  for (size_t i = 0; i < ia[op + mp] - 1; i++) {
    ja[i] += 1;
  }

  for (int i = 0; i < op + mp; i++) {
    for (int j = ia[i] - 1; j < ia[i + 1] - 1; j++) {
      irn[k] = i + 1;
      jcn[k] = ja[j];
      k++;
    }
  }
}

void PNMsolver::Function_DS(double pressure) {
  Ds = (Ds_LIST[6] - Ds_LIST[0]) / (50e6 - 1e6) * (pressure - 1e6) + Ds_LIST[0];
};

double PNMsolver::compre(double pressure) {
  double Tr = Temperature / T_critical;
  double Pr = pressure / P_critical;

  double w_a = 0.45723553;
  double w_b = 0.07779607;
  double w = 0.008;
  double m = 0.37464 + 1.54226 * w - 0.2699 * w * w;

  double A = w_a * pow(1 + m * (1 - sqrt(Tr)), 2) * Pr / pow(Tr, 2);
  double B = w_b * Pr / Tr;

  double solutions[3]{-1e4, -1e4, -1e4};
  gsl_poly_solve_cubic(B - 1, A - 3 * B * B - 2 * B, -A * B + B * B + B * B * B, &solutions[0], &solutions[1], &solutions[2]);

  if ((int)solutions[2] != -1e4) {
    std::sort(solutions, solutions + 3, std::greater<double>());
    return solutions[0];
  } else {
    return solutions[0];
  }
};

double PNMsolver::visco(double p, double z, double T) {
  p = 0.00014504 * p;                                                       // pa -> psi
  T = 1.8 * T;                                                              // k -> Rankin
  double density_of_gas = 28.967 * 0.5537 * p / (z * 10.732 * T) / 62.428;  // g/cm3
  double Mg = 28.967 * 0.5537;
  double X = 3.448 + 986.4 / (T) + 0.001 * Mg;  // T in R, M in g/mol
  double Y = 2.447 - 0.2224 * X;
  double K = (9.379 + 0.02 * Mg) * pow(T, 1.5) / (209.2 + 19.26 * Mg + T);
  return 1e-7 * K * exp(X * pow(density_of_gas, Y));  // cp -> Pa s
};

void PNMsolver::memory() {
  std::vector<std::string> fileList = getFilesInFolder(folderPath);
  bool flag{false};
  for (const auto& file : fileList) {
    if (file.find(string("voxels_number")) != string::npos) {
      ifstream files(file, ios::in);
      if (files.is_open()) {
        flag = true;
      }
      string sline;
      string shead = "=";
      string sshead = ",";
      string ssshead = ";";
      string::size_type idx{0};
      string::size_type idx1{0};
      string::size_type idx2{0};
      vector<int> iputings;
      getline(files, sline);
      assert(idx2 = sline.find(ssshead) != string::npos);
      while ((idx = sline.find(shead, idx)) != string::npos && (idx1 = sline.find(sshead, idx1)) != string::npos) {
        istringstream ss(sline.substr(idx + 1, idx1 - idx - 1));
        int ii;
        ss >> ii;
        iputings.push_back(ii);
        idx++;
        idx1++;
      }
      istringstream ss(sline.substr(idx + 1, idx2 - idx - 1));
      int ii;
      ss >> ii;
      iputings.push_back(ii);

      getline(files, sline);
      getline(files, sline);
      idx = 0;
      idx1 = 0;
      while ((idx = sline.find(shead, idx)) != string::npos && (idx1 = sline.find("\t", idx1)) != string::npos) {
        istringstream ss(sline.substr(idx + 1, idx1 - idx - 1));
        int ii;
        ss >> ii;
        iputings.push_back(ii);
        idx++;
        idx1++;
      }

      while ((idx = sline.find(shead, idx)) != string::npos) {
        istringstream ss(sline.substr(idx + 1));
        int ii;
        ss >> ii;
        iputings.push_back(ii);
        idx++;
        idx1++;
      }

      int num = int(iputings[0]);
      inlet = num;
      num = int(iputings[1]);
      outlet = num;
      num = int(iputings[2]);
      m_inlet = num;
      num = int(iputings[3]);
      m_outlet = num;
      num = int(iputings[4]);
      op = num;
      num = int(iputings[5]);
      mp = num;
      num = int(iputings[6]);
      pn = num;
      num = int(iputings[7]);
      tn = num;
    }
  }

  macro_n = inlet + op + outlet;
  micro_n = m_inlet + mp + m_outlet;
  para_macro = inlet + outlet + m_inlet;

  if (flag == false) {
    cout << "voxel file missed!" << endl;
    abort();
  }

  cout << "pn = " << pn << endl;
  cout << "tn = " << tn << endl;
  cout << "inlet = " << inlet << "; " << "outlet = " << outlet << "; " << "m_inlet = " << m_inlet << "; " << "m_outlet = " << m_outlet << "; " << "op = " << op << "; " << "mp = " << mp << "; "
       << endl;

  Pb = new pore[pn];
  Tb_in = new throat[2 * tn];
  Tb = new throatmerge[2 * tn];
}

void PNMsolver::initial_condition() {
  double start = omp_get_wtime();
  for (int i = 0; i < pn; i++) {
    Pb[i].pressure = inlet_pre;  //- double(double(i) / double(pn) * 100)
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

  for (int i = 0; i < pn; i++) {
    Pb[i].mole_frac_co2 = kong::outlet_co2_mole_farc;  //- double(double(i) / double(pn) * 100)
    Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
  }

  for (int i = 0; i < inlet; i++) {
    Pb[i].mole_frac_co2 = kong::inlet_co2_mole_frac;  //- double(double(i) / double(pn) * 100)
    Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
  }

  for (int i = macro_n; i < macro_n + m_inlet; i++) {
    Pb[i].mole_frac_co2 = kong::inlet_co2_mole_frac;  //- double(double(i) / double(pn) * 100)
    Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
  }

  for (int i = macro_n - outlet; i < macro_n; i++) {
    Pb[i].mole_frac_co2 = kong::outlet_co2_mole_farc;
    Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
  }
  for (int i = pn - m_outlet; i < pn; i++) {
    Pb[i].mole_frac_co2 = kong::outlet_co2_mole_farc;
    Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
  }

  double end = omp_get_wtime();
  printf("initial_condition start = %.16g\tend = %.16g\tdiff = %.16g\n", start, end, end - start);
};

void PNMsolver::initial_condition(int flag) {
  string filename = "CO2_mehante_";
  filename.append(to_string(Time_step));
  ifstream file(filename + ".vtk", ios::in);
  assert(file.is_open());
  string s;
  string head = "LOOKUP_TABLE table3";
  while (getline(file, s)) {
    if (s.find(head) == 0) {
      break;
    }
  }

  for (int i = 0; i < pn; i++) {
    file >> Pb[i].pressure;
    Pb[i].pressure += outlet_pre;
    Pb[i].pressure_old = Pb[i].pressure;
  }
  file.close();
}

void PNMsolver::Paramentinput() {
  cout << "亚分辨区域均质" << endl;
  std::vector<std::string> fileList = getFilesInFolder(folderPath);
  bool flag{false};
  for (const auto& file : fileList) {
    if (file.find(string("full_pore")) != string::npos) {
      ifstream porefile(file, ios::in);
      if (porefile.is_open()) {
        flag = true;
      }

      if (Flag_Hybrid == true) {
        for (int i = 0; i < pn; i++) {
          double waste{0};
          porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >>
              Pb[i].porosity >> Pb[i].km;
          Pb[i].full_coord_ori = Pb[i].full_coord;
          Pb[i].full_accum_ori = Pb[i].full_accum;
          Pb[i].km = ko;
        }
      } else {
        // ofstream inlet_coo("inlet_coo.txt",ios::app);
        for (int i = 0; i < pn; i++) {
          double waste{0};
          porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum;  // REV
          Pb[i].full_coord_ori = Pb[i].full_coord;
          Pb[i].full_accum_ori = Pb[i].full_accum;
          Pb[i].km = ko;

          // if (i < inlet)
          // {
          // 	inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z <<
          // "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          // }
          // else if (i < macro_n + m_inlet && i >= macro_n)
          // {
          // 	inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z <<
          // "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          // }
        }
        porefile.close();

        // inlet_coo.close();
        // int count = 1259;
        // ifstream inlet_coo1("filtered_inlet_coo.txt", ios::in);
        // ostringstream name;
        // name << "filtered_inlet_coo.vtk";
        // ofstream outfile(name.str().c_str());
        // outfile << "# vtk DataFile Version 2.0" << endl;
        // outfile << "output.vtk" << endl;
        // outfile << "ASCII " << endl;
        // outfile << "DATASET POLYDATA " << endl;
        // outfile << "POINTS " << count << " float" << endl;
        // for (size_t i = 0; i < count; i++)
        // {
        // 	double x,y,z,id,r,type;
        // 	inlet_coo1 >> x >> y >> z >> id >> r >> type;
        // 	outfile << x << " " << y << " " << z << endl;
        // }
        // inlet_coo1.close();
        // inlet_coo1.open("inlet_coo.txt", ios::in);
        // outfile << "POINT_DATA "
        // 		<< "\t" << count << endl;
        // outfile << "SCALARS size_pb double 1" << endl;
        // outfile << "LOOKUP_TABLE table1" << endl;
        // for (size_t i = 0; i < count; i++)
        // {
        // 	double x,y,z,id,r,type;
        // 	inlet_coo1 >> x >> y >> z >> id >> r >> type;
        // 	outfile << r << endl;
        // }
        // inlet_coo1.close();

        // inlet_coo1.open("inlet_coo.txt", ios::in);
        // outfile << "SCALARS NUMBER double 1" << endl;
        // outfile << "LOOKUP_TABLE table2" << endl;
        // for (size_t i = 0; i < count; i++)
        // {
        // 	double x,y,z,id,r,type;
        // 	inlet_coo1 >> x >> y >> z >> id >> r >> type;
        // 	outfile << id << endl;
        // }
        // inlet_coo1.close();

        // inlet_coo1.open("inlet_coo.txt", ios::in);
        // outfile << "SCALARS TYPE double 1" << endl;
        // outfile << "LOOKUP_TABLE table3" << endl;
        // for (size_t i = 0; i < count; i++)
        // {
        // 	double x,y,z,id,r,type;
        // 	inlet_coo1 >> x >> y >> z >> id >> r >> type;
        // 	outfile << type << endl;
        // }
        // outfile.close();
        // inlet_coo1.close();
      }
    }
  }
  if (flag == false) {
    cout << "porebody file missed!" << endl;
    abort();
  }

  flag = false;
  for (const auto& file : fileList) {
    if (file.find(string("full_throat")) != string::npos) {
      ifstream throatfile(file, ios::in);
      if (throatfile.is_open()) {
        flag = true;
      }

      if (Flag_Hybrid == true) {
        for (int i = 0; i < 2 * tn; i++) {
          throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
        }
      } else {
        for (int i = 0; i < 2 * tn; i++) {
          throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].Length;
        }
      }
      throatfile.close();
    }
  }
  if (flag == false) {
    cout << "throat file missed!" << endl;
    abort();
  }

  for (int i = 0; i < pn; i++) {
    Pb[i].X = voxel_size * Pb[i].X;
    Pb[i].Y = voxel_size * Pb[i].Y;
    Pb[i].Z = voxel_size * Pb[i].Z;
    Pb[i].Radiu = voxel_size * Pb[i].Radiu;
  }

  for (int i = 0; i < 2 * tn; i++) {
    if (Flag_Hybrid == 1) {
      if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
        Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu;  // pnm部分为喉道的半径
      } else {
        Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu;  // Darcy区的为接触面积
      }
      Tb_in[i].Length = voxel_size * Tb_in[i].Length;
      if (Tb_in[i].Length < voxel_size) {
        Tb_in[i].Length = voxel_size;
      }
      Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
      Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
      Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
    } else {
      Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu;
      Tb_in[i].Length = voxel_size * Tb_in[i].Length - Pb[Tb_in[i].ID_1].Radiu - Pb[Tb_in[i].ID_2].Radiu;
      if (Tb_in[i].Length < voxel_size) {
        Tb_in[i].Length = voxel_size;
      }
    }
  }
}

void PNMsolver::Paramentinput(int i)  // 非均质微孔区域
{
  cout << "亚分辨区域非均质" << endl;
  std::vector<std::string> fileList = getFilesInFolder(folderPath);
  bool flag{false};
  for (const auto& file : fileList) {
    if (file.find(string("full_pore")) != string::npos) {
      ifstream porefile(file, ios::in);
      if (porefile.is_open()) {
        flag = true;
      }
      for (int i = 0; i < pn; i++) {
        double waste{0};
        porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >>
            Pb[i].porosity >> Pb[i].km;
      }
      porefile.close();
    }
  }
  if (flag == false) {
    cout << "porebody file missed!" << endl;
    abort();
  }

  flag = false;
  for (const auto& file : fileList) {
    if (file.find(string("full_throat")) != string::npos) {
      ifstream throatfile(file, ios::in);
      if (throatfile.is_open()) {
        flag = true;
      }

      for (int i = 0; i < 2 * tn; i++) {
        throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
      }
      throatfile.close();
    }
  }
  if (flag == false) {
    cout << "throat file missed!" << endl;
    abort();
  }

  for (int i = 0; i < pn; i++) {
    Pb[i].X = voxel_size * Pb[i].X;
    Pb[i].Y = voxel_size * Pb[i].Y;
    Pb[i].Z = voxel_size * Pb[i].Z;
    Pb[i].Radiu = voxel_size * Pb[i].Radiu;
  }

  for (int i = 0; i < 2 * tn; i++) {
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu;  // pnm部分为喉道的半径
    } else {
      Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu;  // Darcy区的为接触面积
    }
    Tb_in[i].Length = voxel_size * Tb_in[i].Length;
    Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
    Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
    Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
  }
}

void PNMsolver::para_cal() {
  // 计算孔隙的体积
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    if (Pb[i].type == 0) {
      Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;  // 孔隙网络单元
    } else if (Pb[i].type == 1) {
      Pb[i].volume = pow(Pb[i].Radiu, 3);  // 正方形微孔单元
    } else {
      Pb[i].volume = pow(Pb[i].Radiu, 3) / 2;  // 2×2×1、1×2×2和2×1×2的微孔网格
    }
  }

  // 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    // Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 *
    // pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
    Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
    Pb[i].compre_old = Pb[i].compre;
    Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
    Pb[i].visco_old = Pb[i].visco;
  }

  // Total gas content
  double compre_1 = compre(inlet_pre);
  double compre_2 = compre(outlet_pre);
  double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
  double n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_macro)
#endif
  for (int i = inlet; i < macro_n - outlet; i++) {
    total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) -
                        outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
    total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
  }

  total_p = total_macro + total_micro_free + total_micro_ad;
  cout << "total_macro = " << total_macro << endl;
  cout << "total_micro_free = " << total_micro_free << endl;
  cout << "total_micro_ad = " << total_micro_ad << endl;
  cout << "total_p = " << total_p << endl;

  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
    }
  }
  // ofstream Tb_out("Tb_out.txt");
  // for (size_t i = 0; i < 2 * tn; i++)
  // {
  // 	Tb_out << Tb[i].ID_1 << " " << Tb[i].ID_2 << endl;
  // }
  // Tb_out.close();

  // full_coord
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  Pb[0].full_accum = Pb[0].full_coord;

  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("para_cal diff = %.16g\n", end - start);
#endif

  coolist.resize(op + mp);   // 非进出口全配位数
  coolist3.resize(op + mp);  // 非进出口的局部指标
  coolist4.resize(op + mp);  // 非进出口的全局指标
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    int counter{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      int counter{0};
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }

  coolist2.resize(op + mp);  // 非进出口累计全配位数
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < coolist2.size(); i++) {
    std::vector<int>::iterator it = coolist.begin() + i;
    coolist2[i] = accumulate(coolist.begin(), it, 0);
  }

  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[op + mp];
  B = new double[op + mp];

  ia = new int[op + mp + 1];
  ja = new int[NA];

  a = new double[NA];

  COO_A = new Acoo[NA];
}

void PNMsolver::para_cal_in_newton() {
  // 计算压缩系数
  for (int i = 0; i < pn; i++) {
    Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
    Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
  }

  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量

  for (int i = 0; i < 2 * tn; i++) {
    // 计算克努森数
    double Knusen_number{0};
    double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
    double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
    double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
    } else {
      Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
    }
    // 计算滑移项
    double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
    double beta = 4;
    double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
    Tb_in[i].Knusen = Knusen_number;
    Tb_in[i].Slip = Slip;
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
      {
        Tb_in[i].Length = 0.5 * voxel_size;
      }
      Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
      temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
      temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
      Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
    } else {
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
    }
  }

  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  Tb[0].Radiu = Tb_in[0].Radiu;
  Tb[0].Conductivity = Tb_in[0].Conductivity;
  Tb[0].Surface_diff_conduc = Tb_in[0].Surface_diff_conduc;
  Tb[0].Knusen = Tb_in[0].Knusen;
  Tb[0].Slip = Tb_in[0].Slip;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
      Tb[label].Conductivity += Tb_in[i].Conductivity;
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
      Tb[label].Radiu = Tb_in[i].Radiu;
      Tb[label].Conductivity = Tb_in[i].Conductivity;
      Tb[label].Surface_diff_conduc = Tb_in[i].Surface_diff_conduc;
      Tb[label].Knusen = Tb_in[i].Knusen;
      Tb[label].Slip = Tb_in[i].Slip;
    }
  }
}

void PNMsolver::para_cal(double mode) {
  // 计算孔隙的体积
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    if (Pb[i].type == 0) {
      Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;  // 孔隙网络单元
    } else if (Pb[i].type == 1) {
      Pb[i].volume = pow(Pb[i].Radiu, 3);  // 正方形微孔单元
    } else {
      Pb[i].volume = pow(Pb[i].Radiu, 3) / 2;  // 2×2×1、1×2×2和2×1×2的微孔网格
    }
  }

  // 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    // Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 *
    // pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
    Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
    Pb[i].compre_old = Pb[i].compre;
    Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
    Pb[i].visco_old = Pb[i].visco;
  }

  // Total gas content
  double compre_1 = compre(inlet_pre);
  double compre_2 = compre(outlet_pre);
  double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
  double n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_macro)
#endif
  for (int i = inlet; i < macro_n - outlet; i++) {
    total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) -
                        outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
    total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
  }

  total_p = total_macro + total_micro_free + total_micro_ad;
  cout << "total_macro = " << total_macro << endl;
  cout << "total_micro_free = " << total_micro_free << endl;
  cout << "total_micro_ad = " << total_micro_ad << endl;
  cout << "total_p = " << total_p << endl;

  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
                                                                                                          // 计算克努森数
  double Knusen_number{0};
  double Average_pressure{0};
  double Average_compre{0};
  double Average_visco{0};
  double Slip{0};
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) private(Knusen_number, Average_pressure, Average_compre, Average_visco, Slip, temp1, temp2, temp11, temp22, angle1, angle2, length1, length2)
#endif
  for (int i = 0; i < 2 * tn; i++) {
    Knusen_number = 0;
    Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
    Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
    Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
    Slip = 1;
    Tb_in[i].Knusen = Knusen_number;
    Tb_in[i].Slip = 1;
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
      {
        Tb_in[i].Length = 0.5 * voxel_size;
      }
      Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
      temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
      temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
      Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = 0;
    } else {
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
      // cout << temp1 << "\t" << temp2 <<"\t"<< Tb_in[i].Conductivity << endl;
    }
  }
  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  Tb[0].Radiu = Tb_in[0].Radiu;
  Tb[0].Conductivity = Tb_in[0].Conductivity;
  Tb[0].Surface_diff_conduc = Tb_in[0].Surface_diff_conduc;
  Tb[0].Knusen = Tb_in[0].Knusen;
  Tb[0].Slip = Tb_in[0].Slip;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
      Tb[label].Conductivity += Tb_in[i].Conductivity;
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
      Tb[label].Radiu = Tb_in[i].Radiu;
      Tb[label].Conductivity = Tb_in[i].Conductivity;
      Tb[label].Surface_diff_conduc = Tb_in[i].Surface_diff_conduc;
      Tb[label].Knusen = Tb_in[i].Knusen;
      Tb[label].Slip = Tb_in[i].Slip;
    }
  }
  // full_coord
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  Pb[0].full_accum = Pb[0].full_coord;

  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("para_cal diff = %.16g\n", end - start);
#endif
  coolist.resize(op + mp);
  coolist3.resize(op + mp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    int counter{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      int counter{0};
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else {
      }
    }
  }

  coolist2.resize(op + mp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < coolist2.size(); i++) {
    std::vector<int>::iterator it = coolist.begin() + i;
    coolist2[i] = accumulate(coolist.begin(), it, 0);
  }

  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[op + mp];
  B = new double[op + mp];

  ia = new int[op + mp + 1];
  ja = new int[NA];

  a = new double[NA];

  COO_A = new Acoo[NA];
}

void PNMsolver::para_cal_in_newton(double mode) {
  // 计算压缩系数
  for (int i = 0; i < pn; i++) {
    Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
    Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
  }

  // 水力传导系数计算
  double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;  // 两点流量计算中的临时存储变量
  for (int i = 0; i < 2 * tn; i++) {
    // 计算克努森数
    double Knusen_number{0};
    double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
    double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
    double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

    double Slip = 1;
    Tb_in[i].Knusen = Knusen_number;
    Tb_in[i].Slip = 1;
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      if (Tb_in[i].Length <= 0)  // 剔除可能存在的负喉道长度
      {
        Tb_in[i].Length = 0.5 * voxel_size;
      }
      Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
      temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
      temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
    } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
      Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
      Tb_in[i].Surface_diff_conduc = 0;
    } else {
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));
      temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
      temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
      temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      Tb_in[i].Surface_diff_conduc = 0;
    }
  }

  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  Tb[0].Radiu = Tb_in[0].Radiu;
  Tb[0].Conductivity = Tb_in[0].Conductivity;
  Tb[0].Surface_diff_conduc = Tb_in[0].Surface_diff_conduc;
  Tb[0].Knusen = Tb_in[0].Knusen;
  Tb[0].Slip = Tb_in[0].Slip;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
      Tb[label].Conductivity += Tb_in[i].Conductivity;
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
      Tb[label].Radiu = Tb_in[i].Radiu;
      Tb[label].Conductivity = Tb_in[i].Conductivity;
      Tb[label].Surface_diff_conduc = Tb_in[i].Surface_diff_conduc;
      Tb[label].Knusen = Tb_in[i].Knusen;
      Tb[label].Slip = Tb_in[i].Slip;
    }
  }
// full_coord
#pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }
#pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  // full_accum
  Pb[0].full_accum = Pb[0].full_coord;
#pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }
}

void PNMsolver::para_cal_kong() {
  // 计算孔隙的体积
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;  // 孔隙网络单元
  }

  // 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    // Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 *
    // pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
    Pb[i].compre = 1;
    Pb[i].compre_old = 1;
    Pb[i].visco = kong::viscosity;
    Pb[i].visco_old = kong::viscosity;
  }

  // merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  Tb[0].Radiu = Tb_in[0].Radiu;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
      Tb[label].Radiu = Tb_in[i].Radiu;
    }
  }

  // ofstream TB_degub("Tb_debug.txt");
  // for (size_t i = 0; i < 2 * tn; i++)
  // {
  // 	TB_degub << Tb[i].ID_1 << "  " << Tb[i].ID_2 << "  " << endl;
  // }
  // TB_degub.close();

// full_coord
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }

  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  Pb[0].full_accum = Pb[0].full_coord;
  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }

  // ofstream Pb_coord("Pb_coord.txt");
  // for (size_t i = 0; i < pn; i++)
  // {
  // 	Pb_coord << Pb[i].full_coord << "  " << Pb[i].full_accum << "  " <<
  // endl;
  // }
  // Pb_coord.close();

#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("para_cal diff = %.16g\n", end - start);
#endif

  coolist.resize(op + mp);   // 非进出口全配位数
  coolist3.resize(op + mp);  // 非进出口的局部指标
  coolist4.resize(op + mp);  // 非进出口的全局指标
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    int counter{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - inlet] += 1;
        coolist3[i - inlet].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      int counter{0};
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
        coolist[i - para_macro] += 1;
        coolist3[i - para_macro].push_back(counter);
        counter++;
      } else {
        counter++;
      }
    }
  }

  coolist2.resize(op + mp);  // 非进出口累计全配位数
                             // #ifdef _OPENMP
  // #pragma omp parallel for num_threads(int(OMP_PARA))
  // #endif
  // 	for (size_t i = 0; i < coolist2.size(); i++)
  // 	{
  // 		std::vector<int>::iterator it = coolist.begin() + i;
  // 		coolist2[i] = accumulate(coolist.begin(), it, 0);
  // 	}
  // 输出vector到文件
  const std::string filename = "vector_data.txt";
  // writeVectorToFile(coolist2, filename);
  // 从文件读取vector
  std::vector<int> newVec = readVectorFromFile(filename);
  coolist2 = newVec;

  ifstream inlet_coo1("filtered_inlet_coo.txt", ios::in);
  if (!inlet_coo1.is_open()) {
    cout << "inlet_coo1 file not found!" << endl;
    abort();
  }
  for (size_t i = 0; i < 1259; i++) {
    double x, y, z, r;
    int id, type;
    inlet_coo1 >> x >> y >> z >> id >> r >> type;
    inlet_boundary[i] = id;
  }
  inlet_coo1.close();

  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
      reverse_mode<double> Pi, Wi;
      reverse_mode<double>* Pjs;
      reverse_mode<double>* Wjs;
      double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
      double Aij = pi * pow(Tb[j].Radiu, 2);
      if (exists) {
        double vij = 1.29 * 0.01 / 60;                                 // m/s
        Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
      } else {
        double vij = 1.04 * 0.01 / 60;                                 // m/s
        Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
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
        double vij = 1.29 * 0.01 / 60;                                 // m/s
        Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
      } else {
        double vij = 1.04 * 0.01 / 60;                                 // m/s
        Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij / con);  // 更新孔隙压力
      }
    }
  }

  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp) * 2];
  B = new double[(op + mp) * 2];

  ia = new int[(op + mp) * 2 + 1];
  ja = new int[NA * 3];

  a = new double[NA * 3];

  COO_A = new Acoo[NA * 3];
}

double PNMsolver::macro_outlet_flow() {
  double Q_outlet = 0;
  for (int i = macro_n - outlet; i < macro_n; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      double rho{0};
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature);  // rho mol/m^3
      } else {
        rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature);  // rho mol/m^3
      }
      Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho;  // 质量流量 g
    }
  }
  return abs(Q_outlet);
}

double PNMsolver::micro_outlet_free_flow() {
  double Q_outlet = 0;
  for (int i = pn - m_outlet; i < pn; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      double rho{0};
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature);  // rho mol/m^3
      } else {
        rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature);  // rho mol/m^3
      }
      Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho;  // 质量流量 g
    }
  }
  return abs(Q_outlet);
}

double PNMsolver::micro_outlet_ad_flow() {
  double Q_outlet = 0;
  for (int i = pn - m_outlet; i < pn; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      Q_outlet += dt * Tb[j].Surface_diff_conduc *
                  (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) -
                   K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure))) *
                  1000;
    }
  }
  return abs(Q_outlet);
}

double PNMsolver::macro_outlet_Q() {
  double Q_outlet = 0;
  for (int i = 0; i < inlet; i++) {
    for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
      reverse_mode<double> Pi, Wi;
      reverse_mode<double>* Pjs;
      reverse_mode<double>* Wjs;
      Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val());  // 体积流量
    }
  }
  return abs(Q_outlet);
};  // 出口大孔流量

double PNMsolver::micro_outlet_advec_Q() {
  double Q_outlet = 0;
  for (int i = macro_n; i < macro_n + m_inlet; i++) {
    reverse_mode<double> Pi, Wi;
    reverse_mode<double>* Pjs;
    reverse_mode<double>* Wjs;
    for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
      Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val());  // 体积流量
    }
  }
  return abs(Q_outlet);
};  // 出口微孔流量

std::pair<double, double> PNMsolver::area_main_Q() {
  double v_outlet = 0;
  double Q_outlet = 0;

  int icount = 0;
  // ofstream area_main_v("area_main_v.txt", ios::app);
  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
      reverse_mode<double> Pi, Wi;
      reverse_mode<double>* Pjs;
      reverse_mode<double>* Wjs;
      double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
      double Aij = pi * pow(Tb[j].Radiu, 2);
      if (exists) {
        v_outlet += 0;
        Q_outlet += 0;
      } else {
        v_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con) / Aij;  // 体积流量
        Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con);
        icount += 1;
        // area_main_v << j << '\t' << Pb[Tb[j].ID_1].pressure << '\t' <<
        // Pb[Tb[j].ID_2].pressure << '\t' << con << '\t' << Aij << '\t' <<
        // abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con)/Aij <<
        // endl;
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
        v_outlet += 0;
        Q_outlet += 0;
      } else {
        v_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con) / Aij;  // 体积流量
        Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con);
        // area_main_v << j << '\t' << Pb[Tb[j].ID_1].pressure << '\t' <<
        // Pb[Tb[j].ID_2].pressure << '\t' << con << '\t' << Aij << '\t' <<
        // abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con)/Aij <<
        // endl;
        icount += 1;
      }
    }
  }
  // area_main_v << "icount = " << icount << endl;
  // area_main_v.close();
  return make_pair(abs(v_outlet / icount), Q_outlet);
}

std::pair<double, double> PNMsolver::area_side_Q() {
  double Q_outlet = 0;
  double v_outlet = 0;
  // ofstream area_side_v("area_side_v.txt", ios::app);
  int icount = 0;
  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    for (int j = Pb[i].full_accum_ori - Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++) {
      reverse_mode<double> Pi, Wi;
      reverse_mode<double>* Pjs;
      reverse_mode<double>* Wjs;
      double con = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
      double Aij = pi * pow(Tb[j].Radiu, 2);
      if (exists) {
        v_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con) / Aij;
        Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con);
        // area_side_v << j << '\t' << Pb[Tb[j].ID_1].pressure << '\t' <<
        // Pb[Tb[j].ID_2].pressure << '\t' << con << '\t' << Aij << '\t' <<
        // abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con)/Aij <<
        // endl;
        icount += 1;
      } else {
        Q_outlet += 0;  // 体积流量
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
        v_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con) / Aij;
        Q_outlet += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con);
        // area_side_v << j << '\t' << Pb[Tb[j].ID_1].pressure << '\t' <<
        // Pb[Tb[j].ID_2].pressure << '\t' << con << '\t' << Aij << '\t' <<
        // abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * con)/Aij <<
        // endl;
        icount += 1;
      } else {
        Q_outlet += 0;  // 体积流量
      }
    }
  }
  // area_side_v << "icount = " << icount << endl;
  // area_side_v.close();
  return make_pair(abs(v_outlet / icount), Q_outlet);
}

double PNMsolver::micro_outlet_diff_Q() {
  double Q_outlet = 0;
  for (int i = pn - m_outlet; i < pn; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      double average_density =
          ((Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) + (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature)) / 2;
      Q_outlet += Tb[j].Surface_diff_conduc *
                  (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) -
                   K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure))) /
                  (average_density * 16e-3);  // 体积流量
    }
  }
  return abs(Q_outlet);
};  // 出口吸附量

double PNMsolver::Peclet_number() {
  double Peclet{0};
  Peclet = macro_outlet_Q() + micro_outlet_advec_Q();
  Peclet = Peclet * (2000 * voxel_size) / (kong::D_dispersion_macro * pow(1745 * voxel_size, 2) * 0.2);
  return Peclet;
}

double PNMsolver::average_outlet_concentration() {
  double average_c = 0;
  for (int i = macro_n - outlet; i < macro_n; i++) {
    average_c += Pb[i].mole_frac_co2;
  }

  for (int i = pn - m_outlet; i < pn; i++) {
    average_c += Pb[i].mole_frac_co2;
  }
  return abs(average_c / double(outlet + m_outlet) / kong::inlet_co2_mole_frac);
}

double PNMsolver::macro_mass_loss() {
  double macro_mass_loss = 0;
  for (int i = inlet; i < macro_n - outlet; i++) {
    macro_mass_loss += (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) -
                       (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
  }
  return (macro_mass_loss);
}

double PNMsolver::micro_free_mass_loss() {
  double micro_mass_loss = 0;
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));          // kg/m3
    double n_ad_old = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));  // kg/m3

    micro_mass_loss += (porosity - n_ad_old / Rho_ad) * (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) -
                       (porosity - n_ad_new / Rho_ad) * (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
  }
  return (micro_mass_loss);
}

double PNMsolver::micro_ad_mass_loss()  // - outlet_pre * Pb[i].volume *
                                        // (porosity - n_ad2 / Rho_ad) * 16 /
                                        // (compre_2 * 8.314 * Temperature)
{
  double micro_ad_mass_loss = 0;
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
    double n_ad_old = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));

    micro_ad_mass_loss += (n_ad_old - n_ad_new) * 1000 * Pb[i].volume;  // 微孔累计产气质量 单位g
  }
  return (micro_ad_mass_loss);
}

void PNMsolver::AMGXsolver_subroutine_co2_mehane(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx, int nnz_amgx) {
  auto start = high_resolution_clock::now();
  static int icount{0};
  if (icount == 0) {
    AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
    icount += 1;
  } else {
    AMGX_matrix_replace_coefficients(A_amgx, n_amgx, nnz_amgx, a, 0);
  }
  AMGX_solver_setup(solver, A_amgx);
  AMGX_vector_upload(b_amgx, n_amgx, 1, B);
  AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
  AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
  AMGX_vector_download(solution_amgx, dX);

  norm_inf = 0;
  for (size_t i = 0; i < (op + mp) * 2; i++) {
    norm_inf += dX[i] * dX[i];
  }
  norm_inf = sqrt(norm_inf);

  /*--------------------------x(t+dt) = x(t) + dx----------------------*/
  // 更新应力场 浓度场
  for (int i = inlet; i < inlet + op; i++) {
    Pb[i].pressure += dX[i - inlet];
    Pb[i].mole_frac_co2 += dX[i - inlet + op + mp];
  }
  for (int i = op; i < op + mp; i++) {
    Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
    Pb[i + inlet + outlet + m_inlet].mole_frac_co2 += dX[i + op + mp];
  }
  /*--------------------------x(t+dt) = x(t) + dx----------------------*/

  /*-----------------------------边界条件---------------------------------*/
  if (FLAG == 0) {
    // ofstream boundarty("boundary.txt", ios::app);
    // boundarty << " wrong "  << endl;
    for (int i = 0; i < inlet; i++) {
      // Pb[i].pressure += dX[Tb[i].ID_2 - inlet];
      // Pb[i].mole_frac_co2 = inlet_co2_mole_frac;
    }
    for (int i = macro_n; i < macro_n + m_inlet; i++) {
      // Pb[i].pressure = Pb[Tb[Pb[i].full_accum -
      // Pb[i].full_coord].ID_2].pressure; Pb[i].mole_frac_co2 =
      // inlet_co2_mole_frac;
    }
    // boundarty.close();
    FLAG += 1;
  } else {
    // ofstream boundarty("boundary.txt", ios::app);
    // boundarty << " ok "  << endl;
    // boundarty.close();
    // for (int i = 0; i < inlet; i++)
    // {
    // 	bool exists = std::binary_search(inlet_boundary.begin(),
    // inlet_boundary.end(), i); 	for (int j = Pb[i].full_accum_ori -
    // Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++)
    // 	{
    // 		reverse_mode<double> Pi, Wi;
    // 		reverse_mode<double> *Pjs;
    // 		reverse_mode<double> *Wjs;
    // 		double con =  conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i,
    // j).val(); 		double Aij = pi * pow(Tb[j].Radiu, 2); if
    // (exists)
    // 		{
    // 			double vij = 1.29 * 0.01 / 60; // m/s
    // 			Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij /
    // con); // 更新孔隙压力
    // 		}
    // 		else
    // 		{
    // 			double vij = 1.04 * 0.01 / 60; // m/s
    // 			Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij /
    // con); // 更新孔隙压力
    // 		}
    // 	}
    // }
    // for (int i = macro_n; i < macro_n + m_inlet; i++)
    // {
    // 	bool exists = std::binary_search(inlet_boundary.begin(),
    // inlet_boundary.end(), i); 	for (int j = Pb[i].full_accum_ori -
    // Pb[i].full_coord_ori; j < Pb[i].full_accum_ori; j++)
    // 	{
    // 		reverse_mode<double> Pi, Wi;
    // 		reverse_mode<double> *Pjs;
    // 		reverse_mode<double> *Wjs;
    // 		double con =  conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i,
    // j).val(); 		double Aij = pi * pow(Tb[j].Radiu, 2); if
    // (exists)
    // 		{
    // 			double vij = 1.29 * 0.01 / 60; // m/s
    // 			Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij /
    // con); // 更新孔隙压力
    // 		}
    // 		else
    // 		{
    // 			double vij = 1.04 * 0.01 / 60; // m/s
    // 			Pb[i].pressure = Pb[Tb[j].ID_2].pressure + (Aij * vij /
    // con); // 更新孔隙压力
    // 		}
    // 	}
    // }
  }

  for (size_t i = inlet + op; i < inlet + op + outlet; i++) {
    Pb[i].mole_frac_co2 = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].mole_frac_co2;
  }
  for (size_t i = macro_n + m_inlet + mp; i < macro_n + m_inlet + mp + m_outlet; i++) {
    Pb[i].mole_frac_co2 = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].mole_frac_co2;
  }

  /*-----------------------------边界条件---------------------------------*/
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time-consuming = " << duration.count() << " MS" << endl;
}

void PNMsolver::AMGXsolver_subroutine_kong(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx, int nnz_amgx) {}

reverse_mode<double> PNMsolver::func(reverse_mode<double>& Pi, reverse_mode<double>*& Pjs, int Pore_id) {
  reverse_mode<double> RETURN;
  size_t counter{0};
  for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum; j++) {
    if (Tb[j].ID_2 < inlet)  // 大孔进口
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      } else {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      }
      RETURN += Tb[j].Surface_diff_conduc *
                (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
      counter++;
    } else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)  // 大孔出口
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      } else {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      }
      RETURN += Tb[j].Surface_diff_conduc *
                (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
      counter++;
    } else if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)  // 微孔进口边界
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      } else {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      }
      RETURN += Tb[j].Surface_diff_conduc *
                (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
      counter++;
    } else if (Tb[j].ID_2 >= pn - m_outlet)  // 微孔出口边界
    {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      } else {
        RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
      }
      RETURN += Tb[j].Surface_diff_conduc *
                (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
      counter++;
    } else {
      if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
        RETURN += Tb[j].Conductivity * (Pi - Pjs[counter]);
      } else {
        RETURN += Tb[j].Conductivity * (Pi - Pjs[counter]);
      }
      RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pjs[counter] / (1 + K_langmuir * Pjs[counter]));
      counter++;
    }
  }
  return RETURN;
}

bool my_compare(vector<double> a, vector<double> b) {
  if (a[0] != b[0])
    return a[0] > b[0];  // 第一级比较
  else {
    if (a[1] != b[1])
      return a[1] > b[1];  // 如果第一级相同，比较第二级
    else
      return a[2] > b[2];  // 如果第二级仍相同，比较第三级
  }
}

void PNMsolver::Matrix_COO2CSR() {
  int num_rows = 2 * (op + mp);
  int nnz = 3 * NA;
  // int num_rows = op + mp;
  // int nnz = NA;

  qsort(COO_A, nnz, sizeof(coo), sort_by_row);  // sort by row

  // ofstream COOA_OUT("COOA_ad_sorted.txt");

  // for (size_t i = 0; i < nnz; i++)
  // {
  // 	COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val
  // << endl;
  // }

#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < num_rows + 1; i++) {
    ia[i] = 0;
  }

  for (int i = 0; i < nnz; i++) {
    ia[COO_A[i].row + 1]++;
    /*        printf("row:%d,column:%d,value:%f \n", idx_tmp[i].row,
     * idx_tmp[i].col, idx_tmp[i].val);*/
  }
  // prefix-scan
  for (int i = 1; i <= num_rows; i++) {
    ia[i] = ia[i] + ia[i - 1];
    /*        printf("%d \n", rows_offsets[i]);*/
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < nnz; i++) {
    ja[i] = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < num_rows + 1; i++) {
    a[i] = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < nnz; i++) {
    ja[i] = COO_A[i].col;
    a[i] = COO_A[i].val;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("coo2csr diff = %.16g\n", end - start);
#endif
  if (Flag_eigen == true) {
    for (size_t i = 0; i < (op + mp) * 2 + 1; i++) {
      ia[i] += 1;
    }

    for (size_t i = 0; i < ia[(op + mp) * 2]; i++) {
      ja[i] += 1;
    }
  }

  // ofstream ia_out("ia_out_ad.txt");
  // for (size_t i = 0; i < num_rows; i++)
  // {
  // 	ia_out << ia[i] << endl;
  // }

  // ofstream ja_out("ja_out_ad.txt");
  // for (size_t i = 0; i < nnz; i++)
  // {
  // 	ja_out << ja[i] << endl;
  // }

  // ofstream a_out("a_out_ad.txt");
  // for (size_t i = 0; i < nnz; i++)
  // {
  // 	a_out << a[i] << endl;
  // }

  // ofstream Tb_out("tb_out.txt");
  // for (size_t i = 0; i < 2 * tn; i++)
  // {
  // 	Tb_out << Tb_in[i].Conductivity << "  " << Tb_in[i].surface_diff_methane
  // << "  " << Tb_in[i].surface_diff_co2 << endl;
  // }
};

int main(int argc, char** argv) {
  char* buf;
  buf = get_current_dir_name();
  folderPath.assign(buf);
  cout << folderPath << endl;

  PNMsolver Solver;
  // Solver.AMGX_solver_CO2_methane();
  Solver.AMGX_solver_C_kong_PNM();
  /*二氧化碳驱替甲烷*/
}