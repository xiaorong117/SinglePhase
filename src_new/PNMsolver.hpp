#include <vector>
#include "Structures.hpp"

class PNMsolver        // 定义类
{
 public:
  PNMsolver(){};
  
  int conjugateGradient_solver(int iters_, double tol_);

  void memory();        // 动态分配存储器

  void initial_condition();
  void initial_condition(int i);        // 断电继续算

  void Paramentinput();                   // 孔喉数据导入函数声明
  void Paramentinput(int i);              // 微孔非均匀文件读取
  void para_cal();                        // 喉道长度等相关参数计算
  void para_cal_in_newton();              // 在牛顿迭代中计算 克努森数
  void para_cal(double);                  // 喉道长度等相关参数计算
  void para_cal_in_newton(double);        // 在牛顿迭代中计算 克努森数

  void para_cal_co2_methane();        // 喉道长度等相关参数计算
  reverse_mode<double> conductivity_sur_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_bulk_test(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_bulk(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_DISPERSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_mehane_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_SURFACEDIFFUSION(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  vector<reverse_mode<double>> ex_adsorption(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int pore_id);
  vector<double> ex_adsorption(double Pi, double Wi, int pore_id);        // 压缩系数
  vector<reverse_mode<double>> ex_adsorption_pro(reverse_mode<double>& Pi, reverse_mode<double>& Wi, int pore_id);
  vector<double> ex_adsorption_pro(double Pi, double Wi, int pore_id);        // 压缩系数

  reverse_mode<double> conductivity_bulk_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);
  reverse_mode<double> conductivity_co2_DISPERSION_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int pore_id, int throat_id);

  void para_cal_kong();


  void CO2_methane_matrix();
  void kong_matrix();
  void kong_matrix_QIN();
  void kong_matrix_per_QIN();
  void intrinsic_permeability_matrix();
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

  reverse_mode<double> func_TRANSPORT_FLOW_kong(reverse_mode<double>& Pi, reverse_mode<double>* Pjs, reverse_mode<double>& Wi, reverse_mode<double>* Wjs, int i, int num);

  reverse_mode<double> func_append_kong1(reverse_mode<double>& Pi, reverse_mode<double>* Pjs);
  reverse_mode<double> func_append_kong2(reverse_mode<double>& Pi, reverse_mode<double>* Pjs);

  void AMGXsolver_subroutine_co2_mehane(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx,
                                        int nnz_amgx);        // 混合模型方程求解以及变量更新

  void AMGXsolver_subroutine_per(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx,
                                 int nnz_amgx);        // 混合模型方程求解以及变量更新
  void AMGXsolver_subroutine_kong(AMGX_matrix_handle& A_amgx, AMGX_vector_handle& b_amgx, AMGX_vector_handle& solution_amgx, AMGX_solver_handle& solver, int n_amgx, int nnz_amgx);

  double macro_outlet_flow();             // 出口大孔流量
  double macro_outlet_Q();                // 出口大孔流量
  double micro_outlet_free_flow();        // 出口微孔流量
  double micro_outlet_ad_flow();          // 出口吸附量
  double micro_outlet_advec_Q();          // 出口微孔流量
  double micro_outlet_diff_Q();           // 出口吸附量
  array<double, 2> average_outlet_concentration();
  std::pair<double, double> area_main_Q();
  std::pair<double, double> area_side_Q();
  double Peclet_number();
  double macro_mass_loss();
  double micro_free_mass_loss();
  double micro_ad_mass_loss();


  void output_co2_methane(int n);


  void AMGX_solver_CO2_methane();        // 混合模型二氧化碳驱替甲烷
  void AMGX_solver_C_kong_PNM();         // kong

  void AMGX_solver_C_kong_PNM_Neumann_boundary();

  void AMGX_velocity_boundary_incompressible_per();

  void AMGX_Neumann_boundary_QIN_incompressible();

  void EIGEN_GPU_velocity_boundary_incompressible_per();

  void Eigen_subroutine_per(Eigen::SparseMatrix<double, Eigen::RowMajor>&, Eigen::VectorXd&);

 private:

  vector<int> coolist;
  vector<int> coolist2;
  vector<int> coolist5;
  vector<int> coolist6;

  double error;
  int time_step = Time_step;
  double time_all = pyhsic_time;
  double dt;
  double dt2;        // 与dt初值相同，用于输出结果文件
  double Q_outlet_macro;
  double Q_outlet_free_micro;
  double Q_outlet_ad_micro;
  double Q_outlet_REV;
  double leaning_rate;        // 松弛因子

  double free_macro_loss;
  double free_micro_loss;
  double ad_micro_loss;

  double clay_loss;
  double fracture_loss;
  double OM_HP_free_loss;
  double OM_LP_free_loss;
  double OM_HP_ad_loss;
  double OM_LP_ad_loss;

  double total_p;        // total gas content in research domian
  double total_macro;
  double total_micro_free;
  double total_micro_ad;

  double total_clay;
  double total_fracture;
  double total_OM_HP_free;
  double total_OM_LP_free;
  double total_OM_HP_ad;
  double total_OM_LP_ad;

  double norm_inf;
  double eps;            // set residual for dx
  double eps_per;        // set residual for dx

  int iterations_number;

 public:
  ~PNMsolver(){}        // 析构函数，释放动态存储
};