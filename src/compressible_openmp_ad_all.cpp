#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"
#include <Eigen/SPQRSupport>
#include <omp.h>
#include <ctime>
#include <chrono>
#include <numeric>
#include <sys/types.h>
#include <dirent.h>
#include <iterator>
#include <filesystem>
#include <unistd.h> // 函数所在头文件
// For gsl
#include <gsl/gsl_poly.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

// For AMGX
#include <amgx_c.h>
#include <amgx_config.h>

/* Using updated (v2) interfaces to cublas usparseSparseToDense*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>	  // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
// extern "C" {
//   #include "mkl.h"
//   }

#include "fadiff.h"
#include "badiff.h"
#include "mpi.h"
#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654
#define OMP_PARA 20
using namespace std;
using namespace std::chrono;
using namespace fadbad;

template <typename T>
using reverse_mode = B<T>;

typedef struct coo
{
	int row, col;
	double val;
} Acoo;

int sort_by_row(const void *a, const void *b)
{
	if (((Acoo *)a)->row != ((Acoo *)b)->row)
	{
		return ((Acoo *)a)->row > ((Acoo *)b)->row;
	}
	else
	{
		return ((Acoo *)a)->col > ((Acoo *)b)->col;
	}
}

// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
double iters_globa{0};
////常量设置
double pi = 3.1415927;
double gas_vis = 2e-5;				 // 粘度
double porosity = 0.1;				 // 孔隙率
double ko = 1e-21;					 // 微孔达西渗透率 m^2
double inlet_pre = 200;				 // 进口压力 Pa
double outlet_pre = inlet_pre - 100; // 出口压力 Pa
double D = 9e-9;					 // 扩散系数
double Effect_D = 0.05 * D;			 // 微孔中的有效扩散系数
double voxel_size = 32e-9;			 // 像素尺寸，单位m    5.345e-6 8e-9
double domain_size_cubic = 50;
double T_critical{190.564};		// 甲烷的临界温度 190.564K
double P_critical{4.599 * 1e6}; // 甲烷的临界压力 4.599MPa
double Temperature{400};		// 温度
double Rho_ad{400};				// kg/m3
double n_max_ad{44.8};			// kg/m3
double K_langmuir{4e-8};		// Pa^(-1)
double Ds{2.46e-8};				// m2/s
double micro_radius{3.48e-9};
double porosity_HP{0.243};
double porosity_LP{0.081};
double K_OM_HP{1095e-21};
double K_OM_LP{15e-21};

double D_dispersion{2e-5};
double inlet_co2_mole_frac = 0.9;
double outlet_co2_mole_farc = 0.1;
double MOLE_MASS_CO2{0.046};
double MOLE_MASS_CH4{0.016};

double n_max_ad_co2{45};	   // kg/m3
double K_langmuir_co2{1e-7};   // Pa^(-1)
double n_max_ad_ch4{5};		   // kg/m3
double K_langmuir_ch4{1.7e-7}; // Pa^(-1)

vector<double> Ds_LIST({8.32e-9, 9.52e-9, 1.14e-8, 1.44e-8, 1.77e-8, 2.10e-8, 2.46e-8});
int Time_step{0};
int percentage_production_counter{1};
double pyhsic_time{0};
double machine_time{0};
int Flag_eigen{true};
int flag = 2;
int flag1 = 2;
std::string folderPath;

int pn = 1; // 505050不联通 sample3  r=2
int tn = 1;
int inlet = 1, outlet = 1, m_inlet = 1, m_outlet = 1, op = 1, mp = 1;

int macro_n = inlet + op + outlet;
int micro_n = m_inlet + mp + m_outlet;
int para_macro = inlet + outlet + m_inlet;
int NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

double getmax_2(double a, double b)
{
	return a > b ? a : b;
}

double getmax_3(double a, double b, double c)
{
	double temp = getmax_2(a, b);
	temp = getmax_2(temp, c);
	return temp;
}

struct pore
{
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

struct throat
{
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

struct throatmerge
{
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

std::vector<std::string> getFilesInFolder(const std::string &folderPath)
{
	std::vector<std::string> fileList;

	DIR *dir;
	struct dirent *entry;

	// 打开文件夹
	dir = opendir(folderPath.c_str());
	if (dir == nullptr)
	{
		return fileList;
	}

	// 读取文件夹中的文件
	while ((entry = readdir(dir)) != nullptr)
	{
		// 忽略当前目录和上级目录
		if (std::string(entry->d_name) == "." || std::string(entry->d_name) == "..")
		{
			continue;
		}

		// 将文件名添加到列表中
		fileList.push_back(entry->d_name);
	}

	// 关闭文件夹
	closedir(dir);

	return fileList;
}

class PNMsolver // 定义类
{
public:
	double *dX, *B;
	// 求解的时间变量 CSR format
	int *ia, *ja;
	// COO format
	int *irn, *jcn;
	double *a;
	Acoo *COO_A;
	// 申请孔喉的动态存储空间
	pore *Pb;
	throat *Tb_in;
	throatmerge *Tb;
	vector<int> coolist;
	vector<int> coolist2;
	vector<vector<int>> coolist3;
	vector<vector<double>> coolist4;

	double error;
	int time_step = Time_step;
	double time_all = pyhsic_time;
	double dt = 1e-10;
	double dt2 = 1e-8; // 与dt初值相同，用于输出结果文件
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

	double total_p{0}; // total gas content in research domian
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
	double eps = 1e-5;	   // set residual for dx
	double eps_per = 1e-3; // set residual for dx

	int iterations_number = 0;
	// double total_p=2.75554e-8;

	void memory(); // 动态分配存储器
	void initial_condition();
	void initial_condition(int i); // 断电继续算
	void Paramentinput();		   // 孔喉数据导入函数声明
	void Paramentinput(int i);	   // 微孔非均匀文件读取
	void Para_cal_REV();		   //
	void Para_cal_REV_newton();
	void para_cal();				 // 喉道长度等相关参数计算
	void para_cal_in_newton();		 // 在牛顿迭代中计算 克努森数
	void para_cal(double);			 // 喉道长度等相关参数计算
	void para_cal_in_newton(double); // 在牛顿迭代中计算 克努森数
	void para_cal_co2_methane();	 // 喉道长度等相关参数计算
	double compre(double pressure);	 // 压缩系数
	reverse_mode<double> conductivity_sur_test(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	reverse_mode<double> conductivity_bulk_test(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	reverse_mode<double> conductivity_bulk(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	reverse_mode<double> conductivity_co2_DISPERSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	reverse_mode<double> conductivity_mehane_SURFACEDIFFUSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	reverse_mode<double> conductivity_co2_SURFACEDIFFUSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int pore_id, int throat_id);
	vector<reverse_mode<double>> ex_adsorption(reverse_mode<double> &Pi, reverse_mode<double> &Wi, int pore_id);
	vector<double> ex_adsorption(double Pi, double Wi, int pore_id); // 压缩系数
	vector<reverse_mode<double>> ex_adsorption_pro(reverse_mode<double> &Pi, reverse_mode<double> &Wi, int pore_id);
	vector<double> ex_adsorption_pro(double Pi, double Wi, int pore_id); // 压缩系数
	double visco(double pressure, double z, double T);
	double micro_permeability(double pre);
	void Function_DS(double pressure);
	double Function_Slip(double knusen);
	double Function_Slip_clay(double knusen);

	void Eigen_subroutine(Eigen::SparseMatrix<double, Eigen::RowMajor> &, Eigen::VectorXd &); // 非线性求解器
	void PressureMatrix();																	  // 压力矩阵
	double Nor_inf(double A[]);																  // 误差
	void Eigen_solver();																	  // 瞬态扩散迭代求解流程
	void Eigen_solver_per();
	void Eigen_solver_per(double);
	void Eigen_subroutine_per(Eigen::SparseMatrix<double, Eigen::RowMajor> &, Eigen::VectorXd &);

	void AMGXsolver_subroutine_co2_mehane(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx);
	void AMGXsolver_subroutine(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx);
	void AMGXsolver_subroutine_per(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx);
	void AMGXsolver(); // 产气
	void AMGXsplver_REV();
	void PressureMatrix_REV();
	void Matrix_permeability();
	void AMGX_permeability_solver();
	void Matrix_permeability(double);
	void AMGX_permeability_solver(double);
	void CO2_methane_matrix();
	void AMGX_CO2_methane_solver();
	void eigen_CO2_methane_solver();

	void Mysolver_for_inpermeability();
	void Mysolver_for_Appermeability();
	int conjugateGradient_solver(int iters_, double tol_);
	reverse_mode<double> func(reverse_mode<double> &Pi, reverse_mode<double> *&Pjs, int num);
	reverse_mode<double> func_BULK_PHASE_FLOW_in_macro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	reverse_mode<double> func_BULK_PHASE_FLOW_in_micro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	reverse_mode<double> func_BULK_PHASE_FLOW_in_macro_produc(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	reverse_mode<double> func_BULK_PHASE_FLOW_in_micro_produc(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	reverse_mode<double> func_TRANSPORT_FLOW_in_macro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	reverse_mode<double> func_TRANSPORT_FLOW_in_micro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int num);
	void Matrix();
	void Intrinsic_permeability(int myid);

	void CSR2COO();
	void Matrix_COO2CSR();
	void mumps_solver(int);
	void mumps_subroutine(DMUMPS_STRUC_C &, int MYID);
	void mumps_permeability_solver(int MYID, double i);
	void mumps_subroutine_per(DMUMPS_STRUC_C &, int MYID);
	void mumps_permeability_solver(int MYID);

	double macro_outlet_flow();		 // 出口大孔流量
	double micro_outlet_free_flow(); // 出口微孔流量
	double micro_outlet_ad_flow();	 // 出口吸附量
	double macro_outlet_Q();		 // 出口大孔流量
	double micro_outlet_advec_Q();	 // 出口微孔流量
	double micro_outlet_diff_Q();	 // 出口吸附量

	double macro_mass_loss();
	double micro_free_mass_loss();
	double micro_ad_mass_loss();

	double clay_loss_per_step();
	double fracture_loss_per_step();
	double OM_HP_ad_loss_per_step();
	double OM_HP_free_loss_per_step();
	double OM_LP_ad_loss_per_step();
	double OM_LP_free_loss_per_step();

	double REV_OUT();

	void output(int n);			 // 输出VTK文件
	void output(int n, bool m);	 // REV 输出VTK 瞬态
	void output(int n, int m);	 // 渗透率计算输出vtk
	void output(double);		 // 十大攻关输出展示文件
	void output(double, double); // 单重孔网
	void output_co2_methane(int n);

	~PNMsolver() // 析构函数，释放动态存储
	{
		delete[] dX, B;
		// delete[] ia, ja, a;
		delete[] Pb, Tb_in, Tb;
		delete[] COO_A;
	}
};

void PNMsolver::output_co2_methane(int n)
{
	ostringstream name;
	name << "CO2_mehante_" << int(n + 1) << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// 输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table3" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure - outlet_pre << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}

	// 输出压力场信息
	outfile << "SCALARS C_CO2 double 1" << endl;
	outfile << "LOOKUP_TABLE table5" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].mole_frac_co2 << endl;
	}
}

/*
for primary related variables
0 for ex_co2, 1 for ex_ch4, 2 for ad_co2, 3 for ad_ch4, 4 for mass_frac_co2
*/
vector<reverse_mode<double>> PNMsolver::ex_adsorption(reverse_mode<double> &Pi, reverse_mode<double> &Wi, int Pore_id)
{
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
vector<double> PNMsolver::ex_adsorption(double Pi, double Wi, int Pore_id)
{
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
0 for ex_ad ; 1 for ad
*/
vector<reverse_mode<double>> PNMsolver::ex_adsorption_pro(reverse_mode<double> &Pi, reverse_mode<double> &Wi, int Pore_id)
{
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
0 for ex_ad ; 1 for ad
*/
vector<double> PNMsolver::ex_adsorption_pro(double Pi, double Wi, int Pore_id)
{
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

reverse_mode<double> PNMsolver::conductivity_sur_test(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
	// 计算克努森数
	double Knusen_number{0};
	int i = throat_id;

	double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure) / 2;
	double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
	double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
	}
	else
	{
		Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
	}
	// 计算滑移项
	double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
	double beta = 4;
	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	Tb_in[i].Knusen = Knusen_number;
	Tb_in[i].Slip = Slip;
	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
		Tb_in[i].Surface_diff_conduc = 0;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
		temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
		temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		// Tb_in[i].Surface_diff_conduc = 0;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}
		temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

		temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
		temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		// Tb_in[i].Surface_diff_conduc = 0;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Tb_in[i].Conductivity = Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
		Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
	}
	else
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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
	}
	reverse_mode<double> RETURN = Tb_in[i].Surface_diff_conduc;
	return RETURN;
}

reverse_mode<double> PNMsolver::conductivity_bulk_test(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
	// 计算克努森数
	double Knusen_number{0};
	int i = throat_id;

	double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure) / 2;
	double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
	double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
	}
	else
	{
		Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
	}
	// 计算滑移项
	double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
	double beta = 4;
	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	Tb_in[i].Knusen = Knusen_number;
	Tb_in[i].Slip = Slip;
	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
		Tb_in[i].Surface_diff_conduc = 0;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
		temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
		temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		// Tb_in[i].Surface_diff_conduc = 0;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}
		temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

		temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
		temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		// Tb_in[i].Surface_diff_conduc = 0;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Tb_in[i].Conductivity = Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
		Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
	}
	else
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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
	}
	reverse_mode<double> RETURN = Tb_in[i].Conductivity;
	return RETURN;
}

reverse_mode<double> PNMsolver::conductivity_bulk(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
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
	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		Knusen_number1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
		Knusen_number2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
		Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
		alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
		Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
		Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
		Tb_in[i].Slip = (Slip1 + Slip2) / 2;

		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].Conductivity = Tb_in[i].Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
		reverse_mode<double> conductivity = Tb_in[i].Conductivity;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		Knusen_number1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
		Knusen_number2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
		alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
		Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
		Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
		Tb_in[i].Slip = (Slip1 + Slip2) / 2;

		temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp2 = abs(Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
		Tb_in[i].Conductivity = Tb_in[i].Slip * temp1 * temp2 / (temp1 + temp2);
		reverse_mode<double> conductivity = Tb_in[i].Conductivity;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		Knusen_number1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Knusen_number2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
		Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
		alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
		Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
		Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
		Tb_in[i].Slip = (Slip1 + Slip2) / 2;

		temp2 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}
		temp1 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

		Tb_in[i].Conductivity = Tb_in[i].Slip * temp1 * temp2 / (temp1 + temp2);
		reverse_mode<double> conductivity = Tb_in[i].Conductivity;
		return conductivity;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Knusen_number1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Knusen_number2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
		alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
		Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
		Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
		Tb_in[i].Slip = (Slip1 + Slip2) / 2;

		Tb_in[i].Conductivity = Tb_in[i].Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
		reverse_mode<double> conductivity = Tb_in[i].Conductivity;
		return conductivity;
	}
	else
	{
		Knusen_number1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Knusen_number2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		Tb_in[i].Knusen = (Knusen_number1 + Knusen_number2) / 2;
		alpha = 1.358 * 2 / pi * atan(4 * pow(Tb_in[i].Knusen, 0.4));
		Slip1 = (1 + alpha * Knusen_number1) * (1 + beta * Knusen_number1 / (1 + Knusen_number1));
		Slip2 = (1 + alpha * Knusen_number2) * (1 + beta * Knusen_number2 / (1 + Knusen_number2));
		Tb_in[i].Slip = (Slip1 + Slip2) / 2;

		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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

reverse_mode<double> PNMsolver::conductivity_co2_DISPERSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
																										   // 计算克努森数
	double temp_sur_co2_1{0}, temp_sur_co2_2{0};
	double temp_sur_methane_1{0}, temp_sur_methane_2{0};

	int i = throat_id;
	double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].dispersion_coe_co2 = D_dispersion * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp11 = abs(pi * pow(Pb[Tb_in[i].ID_1].Radiu, 1) * D_dispersion);
		temp22 = abs(Tb_in[i].Radiu * D_dispersion * angle2 / length2);
		Tb_in[i].dispersion_coe_co2 = temp11 * temp22 / (temp11 + temp22);
		reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}

		temp11 = abs(Tb_in[i].Radiu * D_dispersion * angle1 / length1);
		temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * D_dispersion);
		Tb_in[i].Conductivity = Tb_in[i].Slip * temp1 * temp2 / (temp1 + temp2);
		Tb_in[i].dispersion_coe_co2 = temp11 * temp22 / (temp11 + temp22);
		reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
		return conductivity;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Tb_in[i].dispersion_coe_co2 = Tb_in[i].Radiu * D_dispersion / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].dispersion_coe_co2;
		return conductivity;
	}
	else
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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

reverse_mode<double> PNMsolver::conductivity_mehane_SURFACEDIFFUSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
																										   // 计算克努森数
	double temp_sur_co2_1{0}, temp_sur_co2_2{0};
	double temp_sur_methane_1{0}, temp_sur_methane_2{0};

	int i = throat_id;
	double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{

		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].surface_diff_methane = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp_sur_methane_1 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);
		temp_sur_methane_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);

		Tb_in[i].surface_diff_methane = temp_sur_methane_1 * temp_sur_methane_2 / (temp_sur_methane_1 + temp_sur_methane_2);
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}
		temp_sur_methane_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
		temp_sur_methane_2 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);

		Tb_in[i].surface_diff_methane = temp_sur_methane_1 * temp_sur_methane_2 / (temp_sur_methane_1 + temp_sur_methane_2);
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
		return conductivity;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Tb_in[i].surface_diff_methane = Tb_in[i].Radiu * Ds / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_methane;
		return conductivity;
	}
	else
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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

reverse_mode<double> PNMsolver::conductivity_co2_SURFACEDIFFUSION(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id, int throat_id)
{
	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
																										   // 计算克努森数
	double temp_sur_co2_1{0}, temp_sur_co2_2{0};
	double temp_sur_methane_1{0}, temp_sur_methane_2{0};

	int i = throat_id;
	double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
	{
		if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
		{
			Tb_in[i].Length = 0.5 * voxel_size;
		}
		Tb_in[i].surface_diff_co2 = Ds * pi * pow(Tb_in[i].Radiu, 2) / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
	{
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
		}
		temp_sur_co2_1 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds);
		temp_sur_co2_2 = abs(Tb_in[i].Radiu * Ds * angle2 / length2);
		Tb_in[i].surface_diff_co2 = temp_sur_co2_1 * temp_sur_co2_2 / (temp_sur_co2_1 + temp_sur_co2_2);
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
		return conductivity;
	}
	else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
		}
		temp_sur_co2_1 = abs(Tb_in[i].Radiu * Ds * angle1 / length1);
		temp_sur_co2_2 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds);
		Tb_in[i].surface_diff_co2 = temp_sur_co2_1 * temp_sur_co2_2 / (temp_sur_co2_1 + temp_sur_co2_2);
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
		return conductivity;
	}
	else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
	{
		Tb_in[i].surface_diff_co2 = Tb_in[i].Radiu * Ds / Tb_in[i].Length;
		reverse_mode<double> conductivity = Tb_in[i].surface_diff_co2;
		return conductivity;
	}
	else
	{
		length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
		length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
		if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
		{
			angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
		}
		else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
		{
			angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
		}
		else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
		{
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

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_micro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
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
	Mass_density_bulk_old = Pb[Pore_id].pressure_old * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);
	Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk - Mass_density_bulk_old) / dt;

	Return += Pb[Pore_id].volume * (ex_adsorption(Pi, Wi, Pore_id)[0] - ex_adsorption(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[0] + ex_adsorption(Pi, Wi, Pore_id)[1] - ex_adsorption(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[1]) / dt;

	/* 流量项 */
	size_t iCounter{0};
	int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Tb_in[j].ID_1].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_co2_SUR = conductivity_co2_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_ch4_SUR = conductivity_mehane_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Conductivity * (Pi - Pjs[iCounter]);
			Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wi, Pore_id)[2]);
			Return += conductivity_ch4_SUR * (ex_adsorption(Pi, Wi, Pore_id)[3] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Pore_id)[3]);
		}
		else
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
			Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_co2_SUR = conductivity_co2_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_ch4_SUR = conductivity_mehane_SURFACEDIFFUSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Conductivity * (Pi - Pjs[iCounter]);
			Return += conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wi, Pore_id)[2]);
			Return += conductivity_ch4_SUR * (ex_adsorption(Pi, Wi, Pore_id)[3] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Pore_id)[3]);
		}
	}

	return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_macro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
	reverse_mode<double> Return{0};
	reverse_mode<double> Mass_density_bulk{0};
	reverse_mode<double> Mass_density_bulk1{0};
	reverse_mode<double> Mass_density_bulk_old{0};
	reverse_mode<double> Mole_mass_bulk{0};
	reverse_mode<double> Mole_mass_bulk1{0};
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
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

			Mole_mass_bulk1 = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
			Mass_density_bulk1 = Pjs[iCounter] * Mole_mass_bulk / (Pb[ID2].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Return += (Mass_density_bulk + Mass_density_bulk1) / 2 * Conductivity * (Pi - Pjs[iCounter]);
		}
		else
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);

			Mole_mass_bulk1 = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
			Mass_density_bulk1 = Pjs[iCounter] * Mole_mass_bulk / (Pb[ID2].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Return += (Mass_density_bulk + Mass_density_bulk1) / 2 * Conductivity * (Pi - Pjs[iCounter]);
		}
	}

	return Return;
}

reverse_mode<double> PNMsolver::func_TRANSPORT_FLOW_in_macro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
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
	Mass_frac_co2_old = ex_adsorption(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[4];
	Return += Pb[Pore_id].volume * (Mass_density_bulk * Mass_frac_co2 - Mass_density_bulk_old * Mass_frac_co2_old) / dt;

	/* 流量项 */
	int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);																// co2 advection term
			Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4]); // co2 dispersion term
		}
		else
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
			Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
			Mass_frac_co2 = ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4];
			Conductivity = conductivity_bulk(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
			Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4]); // co2 dispersion term
		}
	}
	return Return;
}

reverse_mode<double> PNMsolver::func_TRANSPORT_FLOW_in_micro(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
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
	Mass_frac_co2_old = ex_adsorption(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[4];
	Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk * Mass_frac_co2 - Mass_density_bulk_old * Mass_frac_co2_old) / dt;
	Return += Pb[Pore_id].volume * (ex_adsorption(Pi, Wi, Pore_id)[0] - ex_adsorption(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[0]) / dt;

	/* 流量项 */
	int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wi + MOLE_MASS_CH4 * (1 - Wi);
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Pore_id].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);																// co2 advection term
			Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4]); // co2 dispersion term
			Return += Mass_density_bulk * conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[2]);
		}
		else
		{
			Mole_mass_bulk = MOLE_MASS_CO2 * Wjs[iCounter] + MOLE_MASS_CH4 * (1 - Wjs[iCounter]);
			Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
			Mass_frac_co2 = ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4];
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Conductivity_dis_co2 = conductivity_co2_DISPERSION(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_frac_co2 * Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
			Return += Mass_density_bulk * Conductivity_dis_co2 * (ex_adsorption(Pi, Wi, Pore_id)[4] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[4]); // co2 dispersion term
			Return += Mass_density_bulk * conductivity_co2_SUR * (ex_adsorption(Pi, Wi, Pore_id)[2] - ex_adsorption(Pjs[iCounter], Wjs[iCounter], ID2)[2]);
		}
	}
	return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_micro_produc(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
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
	Mass_density_bulk_old = Pb[Pore_id].pressure_old * Mole_mass_bulk_old / (Pb[Pore_id].compre_old * 8.314 * Temperature);
	Return += Pb[Pore_id].volume * porosity * (Mass_density_bulk - Mass_density_bulk_old) / dt;

	Return += Pb[Pore_id].volume * (ex_adsorption_pro(Pi, Wi, Pore_id)[0] - ex_adsorption_pro(Pb[Pore_id].pressure_old, Pb[Pore_id].mole_frac_co2_old, Pore_id)[0]) / dt;

	/* 流量项 */
	size_t iCounter{0};
	int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mass_density_bulk = Pi * Mole_mass_bulk / (Pb[Tb_in[j].ID_1].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_ch4_SUR = conductivity_sur_test(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
			Return += conductivity_ch4_SUR * (ex_adsorption_pro(Pi, Wi, Pore_id)[1] - ex_adsorption_pro(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[1]);
		}
		else
		{
			Mass_density_bulk = Pjs[iCounter] * Mole_mass_bulk / (Pb[Tb_in[j].ID_2].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			conductivity_ch4_SUR = conductivity_sur_test(Pi, Pjs, Wi, Wjs, Pore_id, j);

			Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
			Return += conductivity_ch4_SUR * (ex_adsorption_pro(Pi, Wi, Pore_id)[1] - ex_adsorption_pro(Pjs[iCounter], Wjs[iCounter], Tb_in[j].ID_2)[1]);
		}
	}
	return Return;
}

reverse_mode<double> PNMsolver::func_BULK_PHASE_FLOW_in_macro_produc(reverse_mode<double> &Pi, reverse_mode<double> *Pjs, reverse_mode<double> &Wi, reverse_mode<double> *Wjs, int Pore_id)
{
	reverse_mode<double> Return{0};
	reverse_mode<double> Mass_density_bulk{0};
	reverse_mode<double> Mass_density_bulk_old{0};
	reverse_mode<double> Mole_mass_bulk{0};
	reverse_mode<double> Mole_mass_bulk_old{0};
	reverse_mode<double> Conductivity{0};
	size_t iCounter{0};

	/* 时间项 */
	Mass_density_bulk = Pi * 0.016 / (Pb[Pore_id].compre * 8.314 * Temperature);

	Mass_density_bulk_old = Pb[Pore_id].pressure_old * 0.016 / (Pb[Pore_id].compre_old * 8.314 * Temperature);
	Return += Pb[Pore_id].volume * (Mass_density_bulk - Mass_density_bulk_old) / dt;

	/* 流量项 */
	int ID2 = Tb_in[Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori].ID_2;
	for (int j = Pb[Pore_id].full_accum_ori - Pb[Pore_id].full_coord_ori; j < Pb[Pore_id].full_accum_ori; j++)
	{
		// 判断是否连接同一微孔区域 不用判断是否是进出口 不关于进出口的变量求导就行了
		if (Tb_in[j].ID_2 != ID2)
		{
			iCounter++;
			ID2 = Tb_in[j].ID_2;
		}

		if (Pb[Tb_in[j].ID_1].pressure > Pb[Tb_in[j].ID_2].pressure)
		{
			Mass_density_bulk = Pi * 0.016 / (Pb[Pore_id].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
		}
		else
		{
			Mass_density_bulk = Pjs[iCounter] * 0.016 / (Pb[ID2].compre * 8.314 * Temperature);
			Conductivity = conductivity_bulk_test(Pi, Pjs, Wi, Wjs, Pore_id, j);
			Return += Mass_density_bulk * Conductivity * (Pi - Pjs[iCounter]);
		}
	}
	return Return;
}

void PNMsolver::para_cal_co2_methane() // 喉道长度等相关参数计算
{
	// 计算孔隙的体积
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type == 0)
		{
			Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3; // 孔隙网络单元
		}
		else if (Pb[i].type == 1)
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3); // 正方形微孔单元
		}
		else
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3) / 2; // 2×2×1、1×2×2和2×1×2的微孔网格
		}
	}

	// 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = 1;
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
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
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
		}
		else
		{
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
	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	Pb[0].full_accum = Pb[0].full_coord;

	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}

	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);

	coolist.resize(op + mp);  // 非进出口全配位数
	coolist3.resize(op + mp); // 非进出口的局部指标
	coolist4.resize(op + mp); // 非进出口的全局指标
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		int counter{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else
			{
				counter++;
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			int counter{0};
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else
			{
				counter++;
			}
		}
	}

	coolist2.resize(op + mp); // 非进出口累计全配位数
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < coolist2.size(); i++)
	{
		std::vector<int>::iterator it = coolist.begin() + i;
		coolist2[i] = accumulate(coolist.begin(), it, 0);
	}

	NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
	coolist.clear();
	dX = new double[(op + mp) * 2];
	B = new double[(op + mp) * 2];

	ia = new int[(op + mp) * 2 + 1];
	ja = new int[NA * 4];

	a = new double[NA * 4];

	COO_A = new Acoo[NA * 4];
}

void PNMsolver::CO2_methane_matrix()
{
	/* -------------------------------------------------------------------------------------  */
	/* BULK PHASE EQUATION SOLEVR */
	/* -------------------------------------------------------------------------------------  */

	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	int counter = 0;
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < NA * 3; i++)
	{
		COO_A[i].col = 0;
		COO_A[i].row = 0;
		COO_A[i].val = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < (op + mp) * 2; i++)
	{
		B[i] = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];
		Wjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;
		Wi = Pb[i].mole_frac_co2;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
			Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
		}

		F = func_BULK_PHASE_FLOW_in_macro(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);
		B[i - inlet] = -F.val();
		COO_A[i - inlet].row = i - inlet;
		COO_A[i - inlet].col = i - inlet;
		COO_A[i - inlet].val = Pi.d(0);

		COO_A[i - inlet + NA].row = i - inlet;
		COO_A[i - inlet + NA].col = i - inlet + op + mp;
		COO_A[i - inlet + NA].val = Wi.d(0);

		size_t counter{0};	// 跳过进出口
		size_t counter1{0}; // COOA内存指标
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) // 连接的是大孔
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Wjs[counter].d(0);

				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) // 连接的是微孔
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = Wjs[counter].d(0);

				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

/* -------------------------------------------------------------------------------------  */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------  */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];
		Wjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;
		Wi = Pb[i].mole_frac_co2;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
			Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
		}

		F = func_BULK_PHASE_FLOW_in_micro(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);

		B[i - para_macro] = -F.val();
		COO_A[i - para_macro].row = i - para_macro;
		COO_A[i - para_macro].col = i - para_macro;
		COO_A[i - para_macro].val = Pi.d(0);

		COO_A[i - para_macro + NA].row = i - para_macro;
		COO_A[i - para_macro + NA].col = i - para_macro + op + mp;
		COO_A[i - para_macro + NA].val = Wi.d(0);
		size_t counter{0};	// 跳过进出口
		size_t counter1{0}; // COOA内存指标
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) // 连接的是大孔
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].col = Tb[j].ID_2 - inlet + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) // 连接的是微孔
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].col = Tb[j].ID_2 - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

	/* -------------------------------------------------------------------------------------  */
	/* TRANSPORT EQUATION SOLEVR */
	/* -------------------------------------------------------------------------------------  */

	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	counter = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];
		Wjs = new reverse_mode<double>[Pb[i].full_coord];
		Pi = Pb[i].pressure;
		Wi = Pb[i].mole_frac_co2;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
			Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
		}

		F = func_TRANSPORT_FLOW_in_macro(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);

		B[i - inlet + op + mp] = -F.val();
		COO_A[i - inlet + 2 * NA].row = i - inlet + op + mp;
		COO_A[i - inlet + 2 * NA].col = i - inlet;
		COO_A[i - inlet + 2 * NA].val = Pi.d(0);

		COO_A[i - inlet + 3 * NA].row = i - inlet + op + mp;
		COO_A[i - inlet + 3 * NA].col = i - inlet + op + mp;
		COO_A[i - inlet + 3 * NA].val = Wi.d(0);

		size_t counter{0};
		size_t counter1{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].row = i - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].col = Tb[j].ID_2 - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].row = i - inlet + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

/* -------------------------------------------------------------------------------------  */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------  */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];
		Wjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;
		Wi = Pb[i].mole_frac_co2;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
			Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].mole_frac_co2;
		}

		F = func_TRANSPORT_FLOW_in_macro(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);

		B[i - para_macro + op + mp] = -F.val();
		COO_A[i - para_macro + 2 * NA].row = i - para_macro + op + mp;
		COO_A[i - para_macro + 2 * NA].col = i - para_macro;
		COO_A[i - para_macro + 2 * NA].val = Pi.d(0);

		COO_A[i - para_macro + 3 * NA].row = i - para_macro + op + mp;
		COO_A[i - para_macro + 3 * NA].col = i - para_macro + op + mp;
		COO_A[i - para_macro + 3 * NA].val = Wi.d(0);
		size_t counter{0};
		size_t counter1{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].row = i - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].col = Tb[j].ID_2 - inlet + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = Pjs[counter].d(0);

				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].row = i - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
				COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = Wjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

	double end = omp_get_wtime();
	printf("matrix diff = %.16g\n",
		   end - start);

	// ofstream COOA_OUT("COOA_ad_unsorted.txt");

	// for (size_t i = 0; i < 4 * NA - coolist2[op]; i++)
	// {
	// 	COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val << endl;
	// }

	// ofstream B_OUT("B_OUT.txt");

	// for (size_t i = 0; i < 2 * (op + mp); i++)
	// {
	// 	B_OUT << B[i] << endl;
	// }
}

void PNMsolver::Mysolver_for_Appermeability()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;												// label of output file
	int inter_n{0};											// The interation of outer loop of Newton-raphoon method
	double total_flow = 0;									// accumulation production
	ofstream outfile("Apparent_Permeability_mysolver.txt"); // output permeability;
	Flag_eigen = false;
	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	initial_condition();
	para_cal();
	Matrix_permeability();

	int iter = 200000;
	double re = 1e-3;
	conjugateGradient_solver(iter, re);
	for (size_t i = 2; i < 52; i++)
	{
		para_cal_in_newton();
		Matrix_permeability();
		conjugateGradient_solver(iter, re);
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_permeability();
			conjugateGradient_solver(iter, re);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t" << endl;
		} while (norm_inf > eps_per);

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();
		micro_diff = micro_outlet_diff_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< inlet_pre / 1e6 << "\t"
				<< duration2.count() / 1000 << "s" << "\t"
				<< endl;
		inlet_pre = i * 1e6;
		outlet_pre = inlet_pre - 100;
		Function_DS(inlet_pre);
		initial_condition();
	}
	output(1, 1);
	outfile.close();
}

int PNMsolver::conjugateGradient_solver(int iters_, double tol_)
{
	// 矩阵的显存空间CSR
	int *d_csr_offsets, *d_csr_columns;
	double *d_csr_values;

	clock_t startTime, endTime;
	startTime = clock();

	int N = op + mp;
	int nnz = ia[op + mp];
	const double tol = tol_;
	const int max_iter = iters_;
	cout << "max_iter:" << max_iter << endl;
	double aa, b, na, r0, r1, rr;

	double *x;
	double *d_x, dot;
	double *d_r, *d_p, *d_Ax;
	int k;
	double alpha, beta, alpham1;

	x = (double *)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++)
	{
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	checkCudaErrors(cudaMalloc((void **)&d_csr_columns, nnz * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_csr_offsets, (N + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_csr_values, nnz * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(double)));

	/* Wrap raw data into cuSPARSE generic API objects */
	cusparseSpMatDescr_t matA = NULL;
	checkCudaErrors(cusparseCreateCsr(&matA, N, N, nnz, d_csr_offsets, d_csr_columns, d_csr_values,
									  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	cusparseDnVecDescr_t vecx = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_64F));
	cusparseDnVecDescr_t vecp = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F));
	cusparseDnVecDescr_t vecAx = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_64F));

	/* Initialize problem data */
	cudaMemcpy(d_csr_columns, ja, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_offsets, ia, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_values, a, nnz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, B, N * sizeof(double), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;

	/* Allocate workspace for cuSPARSE */
	size_t bufferSize = 0;
	checkCudaErrors(cusparseSpMV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
		&beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	void *buffer = NULL;
	checkCudaErrors(cudaMalloc(&buffer, bufferSize));

	/* Begin CG */
	checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
								 CUSPARSE_SPMV_ALG_DEFAULT, buffer));
	checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rr));
	checkCudaErrors(cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));
	checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

	k = 1;

	int MYK = 0;
	double RESI = 0;
	while (r1 > tol * tol * rr && k <= max_iter)
	{
		if (k > 1)
		{
			b = r1 / r0;
			cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
			cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
		}
		else
		{
			cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
		}

		checkCudaErrors(cusparseSpMV(
			cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
			&beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
		cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);

		aa = r1 / dot;

		cublasStatus = cublasDaxpy(cublasHandle, N, &aa, d_p, 1, d_x, 1);
		na = -aa;
		cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

		r0 = r1;
		cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		cudaDeviceSynchronize();
		MYK = k;
		RESI = sqrt(r1) / sqrt(rr);
		printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
		k++;
	}
	printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
	endTime = clock();
	cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = x[i];
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	if (matA)
	{
		checkCudaErrors(cusparseDestroySpMat(matA));
	}
	if (vecx)
	{
		checkCudaErrors(cusparseDestroyDnVec(vecx));
	}
	if (vecAx)
	{
		checkCudaErrors(cusparseDestroyDnVec(vecAx));
	}
	if (vecp)
	{
		checkCudaErrors(cusparseDestroyDnVec(vecp));
	}
	free(x);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);
	cudaFree(d_csr_columns);
	cudaFree(d_csr_offsets);
	cudaFree(d_csr_values);
	return 0;
}

void PNMsolver::Mysolver_for_inpermeability()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													// label of output file
	int inter_n{0};												// The interation of outer loop of Newton-raphoon method
	double total_flow = 0;										// accumulation production
	ofstream outfile("Intrinsic_Permeability_mysolver_cg.txt"); // output permeability;
	Flag_eigen = false;
	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	initial_condition();
	para_cal(1);
	Matrix_permeability(1);

	conjugateGradient_solver(1000000, 1e-8);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start1);

	outfile << (macro + micro_advec) * gas_vis * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << "\t"
			<< inlet_pre / 1e6 << "\t"
			<< duration2.count() / 1000 << "s"
			<< endl;
	output(1, 1);
	outfile.close();
};

double PNMsolver::clay_loss_per_step()
{
	double clay_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		if (Pb[i].type == 0)
		{
			clay_loss_per_step += Pb[i].pressure_old * porosity_LP * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - Pb[i].pressure * porosity_LP * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
	}
	return (clay_loss_per_step);
}

double PNMsolver::fracture_loss_per_step()
{
	double fracture_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		if (Pb[i].type == 1)
		{
			fracture_loss_per_step += Pb[i].pressure_old * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
	}
	return (fracture_loss_per_step);
}

double PNMsolver::OM_HP_ad_loss_per_step()
{
	double OM_HP_ad_loss_per_step = 0;

	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old);
		double compre_2 = compre(Pb[i].pressure);
		double n_ad1 = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old));
		double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
		if (Pb[i].type == 1)
		{
			OM_HP_ad_loss_per_step += Pb[i].volume * (1 - porosity_HP) * (n_ad1 - n_ad2) * 1000;
		}
	}
	return (OM_HP_ad_loss_per_step);
}

double PNMsolver::OM_HP_free_loss_per_step()
{
	double OM_HP_free_loss_per_step = 0;

	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old);
		double compre_2 = compre(Pb[i].pressure);
		double n_ad1 = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old));
		double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
		if (Pb[i].type == 1)
		{
			OM_HP_free_loss_per_step += Pb[i].pressure_old * ((1 - porosity_HP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_HP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
		}
	}
	return (OM_HP_free_loss_per_step);
}

double PNMsolver::OM_LP_ad_loss_per_step()
{
	double OM_LP_ad_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old);
		double compre_2 = compre(Pb[i].pressure);
		double n_ad1 = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old));
		double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
		if (Pb[i].type == 1)
		{
			OM_LP_ad_loss_per_step += Pb[i].volume * (1 - porosity_LP) * (n_ad1 - n_ad2) * 1000;
		}
	}
	return (OM_LP_ad_loss_per_step);
}

double PNMsolver::OM_LP_free_loss_per_step()
{
	double OM_LP_free_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old);
		double compre_2 = compre(Pb[i].pressure);
		double n_ad1 = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old));
		double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
		if (Pb[i].type == 1)
		{
			OM_LP_free_loss_per_step += Pb[i].pressure_old * ((1 - porosity_LP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_LP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
		}
	}
	return (OM_LP_free_loss_per_step);
}

double PNMsolver::REV_OUT()
{
	double Q_outlet = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double rho{0};
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				rho = Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
			}
			Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho; // 质量流量 g
		}
	}

	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double rho{0};
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				rho = Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
			}
			Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho; // 质量流量 g
		}
	}
	return abs(Q_outlet);
}

double PNMsolver::Function_Slip(double knusen)
{
	double alpha_om = 1.358 * 2 / pi * atan(4 * pow(knusen, 0.4));
	double beta_om = 4;
	double Slip_om = (1 + alpha_om * knusen) * (1 + beta_om * knusen / (1 + knusen));
	return Slip_om;
}

double PNMsolver::Function_Slip_clay(double knusen)
{
	double alpha_c = 1.5272 * 2 / pi * atan(2.5 * pow(knusen, 0.5));
	double beta_c = 6;
	double Slip_c = (1 + alpha_c * knusen) * (1 + beta_c * knusen / (1 + knusen));
	return Slip_c;
}

void PNMsolver::Para_cal_REV()
{
	// 计算孔隙的体积
	for (int i = 0; i < pn; i++)
	{
		{
			Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3; // 孔隙网络单元
		}
	}

	// 计算压缩系数 气体粘度
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
		Pb[i].visco_old = Pb[i].visco;
	}

	// Total gas content
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(inlet_pre);
		double compre_2 = compre(outlet_pre);

		double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
		double n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));
		if (Pb[i].type == 0)
		{
			total_clay += inlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 1)
		{
			total_fracture += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 2)
		{
			total_OM_HP_free += inlet_pre * ((1 - porosity_HP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * ((1 - porosity_HP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_HP_ad += Pb[i].volume * (1 - porosity_HP) * (n_ad1 - n_ad2) * 1000;
		}
		else if (Pb[i].type == 3)
		{
			total_OM_LP_free += inlet_pre * ((1 - porosity_LP) * (porosity - n_ad1 / Rho_ad) + porosity_LP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * ((1 - porosity_LP) * (porosity - n_ad2 / Rho_ad) + porosity_LP) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_LP_ad += Pb[i].volume * (1 - porosity_LP) * (n_ad1 - n_ad2) * 1000;
		}
	}
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double compre_1 = compre(inlet_pre);
		double compre_2 = compre(outlet_pre);

		double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
		double n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));

		if (Pb[i].type == 0)
		{
			total_clay += inlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 1)
		{
			total_fracture += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 2)
		{
			total_OM_HP_free += inlet_pre * ((1 - porosity_HP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * ((1 - porosity_HP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_HP_ad += Pb[i].volume * (1 - porosity_HP) * (n_ad1 - n_ad2) * 1000;
		}
		else if (Pb[i].type == 3)
		{
			total_OM_LP_free += inlet_pre * ((1 - porosity_LP) * (porosity - n_ad1 / Rho_ad) + porosity_LP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * ((1 - porosity_LP) * (porosity - n_ad2 / Rho_ad) + porosity_LP) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_LP_ad += Pb[i].volume * (1 - porosity_LP) * (n_ad1 - n_ad2) * 1000;
		}
	}
	total_p = total_clay + total_fracture + total_OM_HP_free + total_OM_HP_ad + total_OM_LP_free + total_OM_LP_ad;
	cout << "total_clay = " << total_clay << endl;
	cout << "total_fracture = " << total_fracture << endl;
	cout << "total_OM_HP_free = " << total_OM_HP_free << endl;
	cout << "total_OM_HP_ad = " << total_OM_HP_ad << endl;
	cout << "total_OM_LP_free = " << total_OM_LP_free << endl;
	cout << "total_OM_LP_ad = " << total_OM_LP_ad << endl;
	cout << "total_p = " << total_p << endl;

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
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

		Rho_ID1 = 0.016 * Pb[Tb_in[i].ID_1].pressure / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
		Rho_ID2 = 0.016 * Pb[Tb_in[i].ID_2].pressure / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
		if (Pb[Tb_in[i].ID_1].type == 0) // clay
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_LP;
		}
		else if (Pb[Tb_in[i].ID_1].type == 1) // crack
		{
			Knusen_number_ID1 = 0;
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_1].type == 2) // OM_type1
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (66e-9);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_HP + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_1].pressure, 2) / Rho_ID1;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // OM_type2
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_LP + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_1].pressure, 2) / Rho_ID1;
		}

		if (Pb[Tb_in[i].ID_2].type == 0) // clay
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_LP;
		}
		else if (Pb[Tb_in[i].ID_2].type == 1) // crack
		{
			Knusen_number_ID2 = 0;
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_2].type == 2) // OM_type1
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (66e-9);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_HP + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_2].pressure, 2) / Rho_ID2;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // OM_type2
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_LP + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_2].pressure, 2) / Rho_ID2;
		}

		temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) * Apparent_K_ID1 / Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].Radiu;
		temp2 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) * Apparent_K_ID2 / Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].Radiu;

		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
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

	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}

	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;
	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
}

void PNMsolver::Para_cal_REV_newton()
{
	// 计算压缩系数 气体粘度
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
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

		Rho_ID1 = 0.016 * Pb[Tb_in[i].ID_1].pressure / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
		Rho_ID2 = 0.016 * Pb[Tb_in[i].ID_2].pressure / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
		if (Pb[Tb_in[i].ID_1].type == 0) // clay
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_LP;
		}
		else if (Pb[Tb_in[i].ID_1].type == 1) // crack
		{
			Knusen_number_ID1 = 0;
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_1].type == 2) // OM_type1
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (66e-9);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_HP + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_1].pressure, 2) / Rho_ID1;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // OM_type2
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].pressure * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * K_OM_LP + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_1].pressure, 2) / Rho_ID1;
		}

		if (Pb[Tb_in[i].ID_2].type == 0) // clay
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_LP;
		}
		else if (Pb[Tb_in[i].ID_2].type == 1) // crack
		{
			Knusen_number_ID2 = 0;
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_2].type == 2) // OM_type1
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (66e-9);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_HP + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_2].pressure, 2) / Rho_ID2;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // OM_type2
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].pressure * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (6.96e-9);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * K_OM_LP + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * Pb[Tb_in[i].ID_2].pressure, 2) / Rho_ID2;
		}
		temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 2) * Apparent_K_ID1 / Pb[Tb_in[i].ID_1].visco / Pb[Tb_in[i].ID_1].Radiu;
		temp2 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 2) * Apparent_K_ID2 / Pb[Tb_in[i].ID_2].visco / Pb[Tb_in[i].ID_2].Radiu;

		Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
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

	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}

	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;
	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
}

void PNMsolver::AMGXsplver_REV()
{
	Flag_eigen = false;
	auto start1 = high_resolution_clock::now();
	double acu_clay{0}, acu_fracture{0}, acu_ad_OM_HP{0}, acu_free_OM_HP{0}, acu_ad_OM_LP{0}, acu_free_OM_LP{0};

	int n{1};
	int inter_n{0};								 // The interation of outer loop of Newton-raphoon method
	double total_loss = 0;						 // accumulation production
	ofstream outfile("Transient.txt", ios::app); // output permeability;

	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	if (time_step != 0)
	{
		initial_condition(1);
		n = percentage_production_counter; // label of output file
	}
	else
	{
		initial_condition();
		n = percentage_production_counter; // label of output file
		output(int(-2), bool(1));
	}

	Para_cal_REV();
	PressureMatrix_REV();
	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/FGMRES.json"); // 200

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

	int n_amgx = op + mp;
	int nnz_amgx = ia[op + mp];
	AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
	AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
	AMGX_pin_memory(a, nnz_amgx * sizeof(double));
	AMGX_pin_memory(B, sizeof(double) * n_amgx);
	AMGX_pin_memory(dX, sizeof(double) * n_amgx);

	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 + machine_time << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"

			<< "total_loss / total_p = " << total_loss / total_p << "\t"
			<< "acu_loss_clay / total_p = " << acu_clay / total_p << "\t"
			<< "acu_loss_fracture / total_p = " << acu_fracture / total_p << "\t"
			<< "acu_loss_ad_OM_HP / total_p = " << acu_ad_OM_HP / total_p << "\t"
			<< "acu_loss_free_OM_HP / total_p = " << acu_free_OM_HP / total_p << "\t"
			<< "acu_loss_ad_OM_LP / total_p = " << acu_ad_OM_LP / total_p << "\t"
			<< "acu_loss_free_OM_LP / total_p = " << acu_free_OM_LP / total_p << "\t"

			<< "acu_loss_clay = " << acu_clay << "\t"
			<< "acu_loss_fracture = " << acu_fracture << "\t"
			<< "acu_loss_ad_OM_HP = " << acu_ad_OM_HP << "\t"
			<< "acu_loss_free_OM_HP = " << acu_free_OM_HP << "\t"
			<< "acu_loss_ad_OM_LP = " << acu_ad_OM_LP << "\t"
			<< "acu_loss_free_OM_LP = " << acu_free_OM_LP << "\t"

			<< "mass_conservation_error = " << abs((Q_outlet_REV) - (clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss)) / (clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss) << "\t"

			<< "total_out_rev = " << Q_outlet_REV << "\t"

			<< "clay_loss = " << clay_loss << "\t"
			<< "fracture_loss = " << fracture_loss << "\t"
			<< "OM_HP_free_loss = " << OM_HP_free_loss << "\t"
			<< "OM_HP_ad_loss = " << OM_HP_ad_loss << "\t"
			<< "OM_LP_free_loss = " << OM_LP_free_loss << "\t"
			<< "OM_LP_ad_loss = " << OM_LP_ad_loss << "\t"
			<< "total_mass_loss = " << clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss << "\t"

			<< endl;

	output(time_step - 1, true); // 初始状态
	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	AMGXsolver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	do
	{
		inter_n = 0;
		do
		{
			Para_cal_REV_newton();
			PressureMatrix_REV();
			AMGXsolver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps);

		time_all += dt;
		acu_clay = 0;
		acu_fracture = 0;
		acu_free_OM_HP = 0;
		acu_free_OM_LP = 0;
		acu_ad_OM_HP = 0;
		acu_ad_OM_LP = 0;

		// acumu gas loss
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			double compre_2 = compre(Pb[i].pressure);
			double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));

			if (Pb[i].type == 0)
			{
				acu_clay += inlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * porosity_LP * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 1)
			{
				acu_fracture += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 2)
			{
				acu_free_OM_HP += inlet_pre * ((1 - porosity_HP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_HP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_HP += Pb[i].volume * (1 - porosity_HP) * (n_ad1 - n_ad2) * 1000;
			}
			else if (Pb[i].type == 3)
			{
				acu_free_OM_LP += inlet_pre * ((1 - porosity_LP) * (porosity - n_ad1 / Rho_ad) + porosity_LP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_LP) * (porosity - n_ad2 / Rho_ad) + porosity_LP) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_LP += Pb[i].volume * (1 - porosity_LP) * (n_ad1 - n_ad2) * 1000;
			}
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			double compre_2 = compre(Pb[i].pressure);
			double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			double n_ad2 = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));

			if (Pb[i].type == 0)
			{
				acu_clay += inlet_pre * Pb[i].volume * porosity_LP * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * porosity_LP * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 1)
			{
				acu_fracture += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 2)
			{
				acu_free_OM_HP += inlet_pre * ((1 - porosity_HP) * (porosity - n_ad1 / Rho_ad) + porosity_HP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_HP) * (porosity - n_ad2 / Rho_ad) + porosity_HP) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_HP += Pb[i].volume * (1 - porosity_HP) * (n_ad1 - n_ad2) * 1000;
			}
			else if (Pb[i].type == 3)
			{
				acu_free_OM_LP += inlet_pre * ((1 - porosity_LP) * (porosity - n_ad1 / Rho_ad) + porosity_LP) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * ((1 - porosity_LP) * (porosity - n_ad2 / Rho_ad) + porosity_LP) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_LP += Pb[i].volume * (1 - porosity_LP) * (n_ad1 - n_ad2) * 1000;
			}
		}

		total_loss = acu_clay + acu_fracture + acu_free_OM_HP + acu_free_OM_LP + acu_ad_OM_HP + acu_ad_OM_LP;

		clay_loss = clay_loss_per_step();
		fracture_loss = fracture_loss_per_step();
		OM_HP_free_loss = OM_HP_free_loss_per_step();
		OM_HP_ad_loss = OM_HP_ad_loss_per_step();
		OM_LP_free_loss = OM_LP_free_loss_per_step();
		OM_LP_ad_loss = OM_LP_ad_loss_per_step();

		Q_outlet_REV = REV_OUT();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"

				<< "total_loss / total_p = " << total_loss / total_p << "\t"
				<< "acu_loss_clay / total_p = " << acu_clay / total_p << "\t"
				<< "acu_loss_fracture / total_p = " << acu_fracture / total_p << "\t"
				<< "acu_loss_ad_OM_HP / total_p = " << acu_ad_OM_HP / total_p << "\t"
				<< "acu_loss_free_OM_HP / total_p = " << acu_free_OM_HP / total_p << "\t"
				<< "acu_loss_ad_OM_LP / total_p = " << acu_ad_OM_LP / total_p << "\t"
				<< "acu_loss_free_OM_LP / total_p = " << acu_free_OM_LP / total_p << "\t"

				<< "acu_loss_clay = " << acu_clay << "\t"
				<< "acu_loss_fracture = " << acu_fracture << "\t"
				<< "acu_loss_ad_OM_HP = " << acu_ad_OM_HP << "\t"
				<< "acu_loss_free_OM_HP = " << acu_free_OM_HP << "\t"
				<< "acu_loss_ad_OM_LP = " << acu_ad_OM_LP << "\t"
				<< "acu_loss_free_OM_LP = " << acu_free_OM_LP << "\t"

				<< "mass_conservation_error = " << abs((Q_outlet_REV) - (clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss)) / (clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss) << "\t"

				<< "total_out_rev = " << Q_outlet_REV << "\t"

				<< "clay_loss = " << clay_loss << "\t"
				<< "fracture_loss = " << fracture_loss << "\t"
				<< "OM_HP_free_loss = " << OM_HP_free_loss << "\t"
				<< "OM_HP_ad_loss = " << OM_HP_ad_loss << "\t"
				<< "OM_LP_free_loss = " << OM_LP_free_loss << "\t"
				<< "OM_LP_ad_loss = " << OM_LP_ad_loss << "\t"
				<< "total_mass_loss = " << clay_loss + fracture_loss + OM_HP_free_loss + OM_HP_ad_loss + OM_LP_free_loss + OM_LP_ad_loss << "\t"

				<< endl;

		for (int i = 0; i < pn; i++)
		{
			Pb[i].pressure_old = Pb[i].pressure;
			Pb[i].compre_old = Pb[i].compre;
		}

		// if (inter_n <= 10)
		// {
		// 	dt = dt * 2;
		// }

		if (total_loss / total_p > 0.01 * n)
		{
			output(time_step, true);
			n++;
		}

		time_step++;
	} while (total_loss / total_p < 0.99);
	output(time_step, true);
	// total_flow / total_p < 0.99
	// while (time_step<100);  while (Error() > 1e-8);
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

void PNMsolver::eigen_CO2_methane_solver()
{
	Flag_eigen = true;
	auto start1 = high_resolution_clock::now();
	double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

	int n{1};
	int inter_n{0};								   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;						   // accumulation production
	ofstream outfile("CO2_methane.txt", ios::app); // output permeability;

	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	if (time_step != 0)
	{
		initial_condition(1);
		n = percentage_production_counter; // label of output file
	}
	else
	{
		initial_condition();
	}
	para_cal_co2_methane();
	CO2_methane_matrix();
	Matrix_COO2CSR();

	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 + machine_time << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"
			<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
			<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
			<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
			<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"
			<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"
			<< "mass_conservation_error = " << 0 << "\t"
			<< "macro_loss = " << free_macro_loss << "\t"
			<< "free_micro_loss = " << free_micro_loss << "\t"
			<< "ad_micro_loss = " << ad_micro_loss << "\t"
			<< "acu_flow_macro = " << acu_flow_macro << "\t"
			<< "acu_free_micro = " << acu_free_micro << "\t"
			<< "acu_ad_micro = " << acu_ad_micro << "\t"
			<< "total_flow / total_p = " << total_flow / total_p << "\t"
			<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
			<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
			<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
			<< endl;
	output_co2_methane(time_step - 1); // 初始状态
	Eigen::SparseMatrix<double, Eigen::RowMajor> A0((op + mp) * 2, (op + mp) * 2);
	Eigen::VectorXd B0((op + mp) * 2, 1);
	Eigen_subroutine(A0, B0);
	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	do
	{
		inter_n = 0;
		do
		{
			CO2_methane_matrix();
			Matrix_COO2CSR();
			Eigen_subroutine(A0, B0);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			acu_flow_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre);
			double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
			double n_ad_old = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			acu_free_micro += (porosity - n_ad_old / Rho_ad) * inlet_pre * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre) - K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)); // 微孔累计产气质量 单位g
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
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"
				<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
				<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
				<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
				<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

				<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

				<< "mass_conservation_error = " << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
				<< "macro_loss = " << free_macro_loss << "\t"
				<< "free_micro_loss = " << free_micro_loss << "\t"
				<< "ad_micro_loss = " << ad_micro_loss << "\t"

				<< "acu_flow_macro = " << acu_flow_macro << "\t"
				<< "acu_free_micro = " << acu_free_micro << "\t"
				<< "acu_ad_micro = " << acu_ad_micro << "\t"
				<< "total_flow / total_p = " << total_flow / total_p << "\t"
				<< "acu_flow_macro / total_sub-resolution poresp = " << acu_flow_macro / total_p << "\t"
				<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
				<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
				<< "eps = " << eps << "\t"
				<< endl;

		for (int i = 0; i < pn; i++)
		{
			Pb[i].pressure_old = Pb[i].pressure;
			Pb[i].compre_old = Pb[i].compre;
			Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
		}

		if (inter_n < 5)
		{
			dt = dt * 2;
		}

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
}

void PNMsolver::AMGX_CO2_methane_solver()
{
	Flag_eigen = false;
	auto start1 = high_resolution_clock::now();
	double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

	int n{1};
	int inter_n{0};								   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;						   // accumulation production
	ofstream outfile("CO2_methane.txt", ios::app); // output permeability;

	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	if (time_step != 0)
	{
		initial_condition(1);
		n = percentage_production_counter; // label of output file
	}
	else
	{
		initial_condition();
	}
	para_cal_co2_methane();
	CO2_methane_matrix();
	Matrix_COO2CSR();
	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/1.json"); // 200

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
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 + machine_time << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"
			<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
			<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
			<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
			<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"
			<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"
			<< "mass_conservation_error = " << 0 << "\t"
			<< "macro_loss = " << free_macro_loss << "\t"
			<< "free_micro_loss = " << free_micro_loss << "\t"
			<< "ad_micro_loss = " << ad_micro_loss << "\t"
			<< "acu_flow_macro = " << acu_flow_macro << "\t"
			<< "acu_free_micro = " << acu_free_micro << "\t"
			<< "acu_ad_micro = " << acu_ad_micro << "\t"
			<< "total_flow / total_p = " << total_flow / total_p << "\t"
			<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
			<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
			<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
			<< endl;
	output_co2_methane(time_step - 1); // 初始状态
	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	do
	{
		inter_n = 0;
		do
		{
			CO2_methane_matrix();
			Matrix_COO2CSR();
			AMGXsolver_subroutine_co2_mehane(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			acu_flow_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre);
			double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
			double n_ad_old = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			acu_free_micro += (porosity - n_ad_old / Rho_ad) * inlet_pre * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre) - K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)); // 微孔累计产气质量 单位g
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
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"
				<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
				<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
				<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
				<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

				<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

				<< "mass_conservation_error = " << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
				<< "macro_loss = " << free_macro_loss << "\t"
				<< "free_micro_loss = " << free_micro_loss << "\t"
				<< "ad_micro_loss = " << ad_micro_loss << "\t"

				<< "acu_flow_macro = " << acu_flow_macro << "\t"
				<< "acu_free_micro = " << acu_free_micro << "\t"
				<< "acu_ad_micro = " << acu_ad_micro << "\t"
				<< "total_flow / total_p = " << total_flow / total_p << "\t"
				<< "acu_flow_macro / total_sub-resolution poresp = " << acu_flow_macro / total_p << "\t"
				<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
				<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
				<< "eps = " << eps << "\t"
				<< endl;

		for (int i = 0; i < pn; i++)
		{
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

void PNMsolver::PressureMatrix_REV()
{
	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	int temp2 = 0;

	ia[0] = 1;
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}

	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}

	/* -------------------------------------------------------------------------------------  */
	/* 粘土0和裂缝1 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = inlet; i < op + inlet; i++)
	{
		if (Pb[i].type == 0)
		{
			B[i - inlet] = porosity_LP * (-0.016) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 1)
		{
			B[i - inlet] = (-0.016) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 2)
		{
			B[i - para_macro] = -0.016 * (porosity_HP + (1 - porosity_HP) * porosity) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - porosity_HP) * Pb[i].volume / dt * ((1 - 0.016 * Pb[i].pressure / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)) - (1 - 0.016 * Pb[i].pressure_old / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)));
		}
		else if (Pb[i].type == 3)
		{
			B[i - para_macro] = -0.016 * (porosity_LP + (1 - porosity_LP) * porosity) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - porosity_LP) * Pb[i].volume / dt * ((1 - 0.016 * Pb[i].pressure / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)) - (1 - 0.016 * Pb[i].pressure_old / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)));
		}

		temp = 0, temp1 = 0;
		num = ia[i - inlet];
		// macropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet) // 进口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}

				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - inlet + 1] = num1 + 1;		// 前i行累计的非零值数量，其中1为ia[0]的值
		ja[num + temp - 1] = i - inlet + 1; // 第i行对角线的值的位置

		if (Pb[i].type == 0)
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume * porosity_LP / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 1)
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 2)
		{
			a[num + temp - 1] = 0.016 * (porosity_HP + (1 - porosity_HP) * porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - porosity_HP) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * Pb[i].pressure - n_max_ad * 0.016 * pow(K_langmuir * Pb[i].pressure, 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * Pb[i].pressure), 2));
		}
		else if (Pb[i].type == 3)
		{
			a[num + temp - 1] = 0.016 * (porosity_LP + (1 - porosity_LP) * porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - porosity_LP) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * Pb[i].pressure - n_max_ad * 0.016 * pow(K_langmuir * Pb[i].pressure, 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * Pb[i].pressure), 2));
		}

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}
					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	/* -------------------------------------------------------------------------------------  */
	/* 有机质两类组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		if (Pb[i].type == 0)
		{
			B[i - inlet] = porosity_LP * (-0.016) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 1)
		{
			B[i - inlet] = (-0.016) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 2)
		{
			B[i - para_macro] = -0.016 * (porosity_HP + (1 - porosity_HP) * porosity) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - porosity_HP) * Pb[i].volume / dt * ((1 - 0.016 * Pb[i].pressure / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)) - (1 - 0.016 * Pb[i].pressure_old / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)));
		}
		else if (Pb[i].type == 3)
		{
			B[i - para_macro] = -0.016 * (porosity_LP + (1 - porosity_LP) * porosity) * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - porosity_LP) * Pb[i].volume / dt * ((1 - 0.016 * Pb[i].pressure / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)) - (1 - 0.016 * Pb[i].pressure_old / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)));
		}

		temp = 0, temp1 = 0;
		num = ia[i - para_macro];
		// micropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进出口边界
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}

				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - para_macro + 1] = num1 + 1;		 // 前i行累计的非零值数量，其中1为ia[0]的值
		ja[num + temp - 1] = i - para_macro + 1; // 第i行对角线的值的位置
		if (Pb[i].type == 0)
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume * porosity_LP / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 1)
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 2)
		{
			a[num + temp - 1] = 0.016 * (porosity_HP + (1 - porosity_HP) * porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - porosity_HP) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * Pb[i].pressure - n_max_ad * 0.016 * pow(K_langmuir * Pb[i].pressure, 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * Pb[i].pressure), 2));
		}
		else if (Pb[i].type == 3)
		{
			a[num + temp - 1] = 0.016 * (porosity_LP + (1 - porosity_LP) * porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - porosity_LP) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * Pb[i].pressure - n_max_ad * 0.016 * pow(K_langmuir * Pb[i].pressure, 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * Pb[i].pressure), 2));
		}

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 主对角线的初始值
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	if (Flag_eigen == false)
	{
		for (size_t i = 0; i < op + mp + 1; i++)
		{
			ia[i] += -1;
		}

		for (size_t i = 0; i < ia[op + mp]; i++)
		{
			ja[i] += -1;
		}
	}
}

void PNMsolver::mumps_permeability_solver(int MYID)
{
	Flag_eigen = {false};
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;											 // label of output file
	int inter_n{0};										 // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;								 // accumulation production
	ofstream outfile("Apparent_Permeability_mumps.txt"); // output permeability;

	memory();
	Paramentinput();
	initial_condition();
	para_cal();
	Matrix_permeability();
	CSR2COO();
	/* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
	DMUMPS_STRUC_C id;
	MUMPS_INT nn = op + mp;
	MUMPS_INT8 nnz = ia[op + mp] - 1;
	id.comm_fortran = USE_COMM_WORLD;
	id.par = 1;
	id.sym = 0;
	id.job = JOB_INIT;
	dmumps_c(&id);

	/* Define the problem on the host */
	if (MYID == 0)
	{
		id.n = nn;
		id.nnz = nnz;
		id.irn = irn;
		id.jcn = jcn;
		id.a = a;
		id.rhs = B;
	}
#define ICNTL(I) icntl[(I) - 1] /* macro s.t. indices match documentation */
	/* No outputs */
	id.ICNTL(1) = 6;
	id.ICNTL(2) = 0;
	id.ICNTL(3) = 6;
	id.ICNTL(4) = 2;
	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 << "\t"
			<< endl;
	mumps_subroutine_per(id, MYID);
	for (size_t i = 2; i < 52; i++)
	{
		para_cal_in_newton();
		Matrix_permeability();
		CSR2COO();
		mumps_subroutine_per(id, MYID);
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_permeability();
			CSR2COO();
			mumps_subroutine_per(id, MYID);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step << endl;
			cout << endl;
		} while (norm_inf > eps);

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();
		micro_diff = micro_outlet_diff_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< inlet_pre / 1e6
				<< endl;
		inlet_pre = i * 1e6;
		outlet_pre = inlet_pre - 100;
		Function_DS(inlet_pre);
		initial_condition();
	}
	output(1, 1);
	outfile.close();
	/* Terminate instance. */
	id.job = JOB_END;
	dmumps_c(&id);
	if (MYID == 0)
	{
		if (!error)
		{
			printf("Nothing wrong");
		}
		else
		{
			printf("An error has occured, please check error code returned by MUMPS.\n");
		}
	}
	MPI_Finalize();
}

void PNMsolver::Eigen_solver_per(double i)
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													   // label of output file
	int inter_n{0};												   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;										   // accumulation production
	ofstream outfile("Intrinsic_Permeability_Eigen_BiCGSTAB.txt"); // output permeability;
	Flag_eigen = true;
	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	initial_condition();
	para_cal(1);
	Matrix();
	Eigen_solver_per(1);

	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine_per(A0, B0);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start1);

	outfile << (macro + micro_advec) * gas_vis * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << "\t"
			<< inlet_pre << "\t"
			<< duration2.count() / 1000 << "s"
			<< endl;
	output(1, 1);
	outfile.close();
}

void PNMsolver::Intrinsic_permeability(int MYID)
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;			   // label of output file
	int inter_n{0};		   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0; // accumulation production

	memory();
	Paramentinput(1);
	initial_condition();
	para_cal(1);

	/*MUMPS*/
	if (flag == 1)
	{
		Flag_eigen = false;
		Matrix();
		CSR2COO();
		/* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
		DMUMPS_STRUC_C id;
		MUMPS_INT nn = op + mp;
		MUMPS_INT8 nnz = ia[op + mp] - 1;
		id.comm_fortran = USE_COMM_WORLD;
		id.par = 1;
		string filename("Intrinsic_permeability_AXB_mumups_");
		if (flag1 == 1)
		{
			id.sym = 0;
			filename += "LU_UNSYM";
			filename += to_string(id.sym);
		}
		else if (flag1 == 2)
		{
			id.sym = 1;
			filename += "LDLT_SPD";
			filename += to_string(id.sym);
		}
		else if (flag1 == 3)
		{
			id.sym = 2;
			filename += "LDLT_general_sym";
			filename += to_string(id.sym);
		}
		id.job = JOB_INIT;
		dmumps_c(&id);

		ofstream outfile(filename + ".txt");
		/* Define the problem on the host */
		if (MYID == 0)
		{
			id.n = nn;
			id.nnz = nnz;
			id.irn = irn;
			id.jcn = jcn;
			id.a = a;
			id.rhs = B;
		}
#define ICNTL(I) icntl[(I) - 1] /* macro s.t. indices match documentation */
#define CNTL(I) cntl[(I) - 1]	/* macro s.t. indices match documentation */
		/* No outputs */
		id.CNTL(1) = 100;
		id.ICNTL(1) = 6;
		id.ICNTL(2) = 0;
		id.ICNTL(3) = 6;
		id.ICNTL(4) = 2;
		/* Call the MUMPS package (analyse, factorization and solve). */
		id.job = 6;
		dmumps_c(&id);
		if (id.infog[0] < 0)
		{
			printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
				   MYID, id.infog[0], id.infog[1]);
			error = 1;
		}

		for (size_t i = 0; i < op + mp; i++)
		{
			dX[i] = B[i];
		}
		/*--------------------------x(t+dt) = x(t) + dx----------------------*/
		// 更新应力场
		for (int i = inlet; i < inlet + op; i++)
		{
			Pb[i].pressure = dX[i - inlet];
		}
		for (int i = op; i < op + mp; i++)
		{
			Pb[i + inlet + outlet + m_inlet].pressure = dX[i];
		}
		/*--------------------------x(t+dt) = x(t) + dx----------------------*/
		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << "machine_time = " << duration2.count() / 1000 << "\t"
				<< "macro_outlet_Q = " << macro << "\t"
				<< "micro_advec = " << micro_advec << "\t"
				<< "permeability = " << (macro + micro_advec) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << endl;
		output(1, 1);
		outfile.close();

		/* Terminate instance. */
		id.job = JOB_END;
		dmumps_c(&id);
		if (MYID == 0)
		{
			if (!error)
			{
				printf("Nothing wrong");
			}
			else
			{
				printf("An error has occured, please check error code returned by MUMPS.\n");
			}
		}
		MPI_Finalize();
	}
	/*MUMPS*/

	// /*EIGEN*/
	if (flag == 2)
	{
		Flag_eigen = true;
		Matrix();
		using namespace Eigen;
		Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
		Eigen::VectorXd B0(op + mp, 1);
		auto start = high_resolution_clock::now();

		for (int i = 0; i < op + mp; ++i)
		{
			for (int j = ia[i]; j < ia[i + 1]; j++)
			{
				A0.coeffRef(i, ja[j - 1] - 1) = a[j - 1];
			}
		}

		for (int i = 0; i < op + mp; i++)
		{
			B0[i] = B[i];
		}

		string filename("Intrinsic_permeability_eigen_");
		if (flag1 == 1)
		{
			filename += "LU";
			SimplicialLLT<SparseMatrix<double, RowMajor>> solver;
			solver.compute(A0);
			VectorXd x = solver.solve(B0);
			for (int i = inlet; i < inlet + op; i++)
			{
				Pb[i].pressure = x[i - inlet];
			}
			for (int i = op; i < op + mp; i++)
			{
				Pb[i + inlet + outlet + m_inlet].pressure = x[i];
			}
		}
		else if (flag1 == 2)
		{
			filename += "LDLT";
			SimplicialLLT<SparseMatrix<double, RowMajor>> solver;
			solver.compute(A0);
			VectorXd x = solver.solve(B0);
			for (int i = inlet; i < inlet + op; i++)
			{
				Pb[i].pressure = x[i - inlet];
			}
			for (int i = op; i < op + mp; i++)
			{
				Pb[i + inlet + outlet + m_inlet].pressure = x[i];
			}
		}
		else if (flag1 == 3)
		{
			filename += "LLT";
			SimplicialLLT<SparseMatrix<double, RowMajor>> solver;
			solver.compute(A0);
			VectorXd x = solver.solve(B0);
			for (int i = inlet; i < inlet + op; i++)
			{
				Pb[i].pressure = x[i - inlet];
			}
			for (int i = op; i < op + mp; i++)
			{
				Pb[i + inlet + outlet + m_inlet].pressure = x[i];
			}
		}
		else if (flag1 == 4)
		{
			filename += "SVD";
			SparseQR<SparseMatrix<double>, AMDOrdering<int>> solver;
			solver.compute(A0);
			VectorXd x = solver.solve(B0);
			for (int i = inlet; i < inlet + op; i++)
			{
				Pb[i].pressure = x[i - inlet];
			}
			for (int i = op; i < op + mp; i++)
			{
				Pb[i + inlet + outlet + m_inlet].pressure = x[i];
			}
		}

		ofstream outfile(filename + ".txt");

		// solver.setTolerance(pow(10, -8));
		// solver.setMaxIterations(5000);
		/*solver.setMaxIterations(3);*/

		// std::cout << "#iterations:     " << solver.iterations() << std::endl;
		// iterations_number = solver.iterations();
		// std::cout << "estimated error: " << solver.error() << std::endl;
		// 更新应力场

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);
		cout << "Time-consuming = " << duration.count() << " MS" << endl;

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << "machine_time = " << duration2.count() / 1000 << "\t"
				<< "macro_outlet_Q = " << macro << "\t"
				<< "micro_advec = " << micro_advec << "\t"
				<< "permeability = " << (macro + micro_advec) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << endl;
		output(1, 1);
		outfile.close();
	}
	/*EIGEN*/

	/*amgx*/
	// begin AMGX initialization
	// AMGX_initialize();

	// AMGX_config_handle config;
	// AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/IDRMSYNC_DILU.json");

	// AMGX_resources_handle rsrc;
	// AMGX_resources_create_simple(&rsrc, config);

	// AMGX_solver_handle solver;
	// AMGX_matrix_handle A_amgx;
	// AMGX_vector_handle b_amgx;
	// AMGX_vector_handle solution_amgx;

	// AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, config);
	// AMGX_matrix_create(&A_amgx, rsrc, AMGX_mode_dDDI);
	// AMGX_vector_create(&b_amgx, rsrc, AMGX_mode_dDDI);
	// AMGX_vector_create(&solution_amgx, rsrc, AMGX_mode_dDDI);

	// int n_amgx = op + mp;
	// int nnz_amgx = ia[op + mp];
	// AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
	// AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
	// AMGX_pin_memory(a, nnz_amgx * sizeof(double));
	// AMGX_pin_memory(B, sizeof(double) * n_amgx);
	// AMGX_pin_memory(dX, sizeof(double) * n_amgx);

	// auto stop3 = high_resolution_clock::now();
	// auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	// outfile << "inner loop = " << inter_n << "\t"
	// 		<< "machine_time = " << duration3.count() / 1000 << "\t"
	// 		<< endl;
	// // end AMGX initialization

	// // ************ begin AMGX solver ************
	// auto start = high_resolution_clock::now();
	// AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
	// AMGX_solver_setup(solver, A_amgx);
	// AMGX_vector_upload(b_amgx, n_amgx, 1, B);
	// AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	// AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
	// AMGX_vector_download(solution_amgx, dX);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场
	// for (int i = inlet; i < inlet + op; i++)
	// {
	// 	Pb[i].pressure = dX[i - inlet];
	// }
	// for (int i = op; i < op + mp; i++)
	// {
	// 	Pb[i + inlet + outlet + m_inlet].pressure = dX[i];
	// }
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	/***********************销毁AMGX***************************/
	// AMGX_unpin_memory(ia);
	// AMGX_unpin_memory(ja);
	// AMGX_unpin_memory(a);
	// AMGX_unpin_memory(B);
	// AMGX_unpin_memory(dX);

	// AMGX_solver_destroy(solver);
	// AMGX_vector_destroy(b_amgx);
	// AMGX_vector_destroy(solution_amgx);
	// AMGX_matrix_destroy(A_amgx);
	// AMGX_resources_destroy(rsrc);
	// AMGX_config_destroy(config);
	// AMGX_finalize();
	// ************ end AMGX solver ************
}

void PNMsolver::Matrix()
{
	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	int temp2 = 0;

	ia[0] = 1;
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}

	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}

	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = inlet; i < op + inlet; i++)
	{
		B[i - inlet] = 0;
		temp = 0, temp1 = 0;
		num = ia[i - inlet];
		// macropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet) // 进口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += 0;
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += 0;
				}

				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - inlet + 1] = num1 + 1; // 前i行累计的非零值数量，其中1为ia[0]的值

		ja[num + temp - 1] = i - inlet + 1; // 第i行对角线的值的位置
		a[num + temp - 1] = 0;				// 主对角线的初始值

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num - 1] += -Tb[j].Conductivity;
					}
					else
					{
						a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num - 1] += -Tb[j].Conductivity;
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num] += -Tb[j].Conductivity;
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num] += -Tb[j].Conductivity;
					}
					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	/* -------------------------------------------------------------------------------------  */
	/* 微孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		B[i - para_macro] = 0;

		temp = 0, temp1 = 0;
		num = ia[i - para_macro];
		// micropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进出口边界
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += 0;
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += 0;
				}
				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - para_macro + 1] = num1 + 1;		 // 前i行累计的非零值数量，其中1为ia[0]的值
		ja[num + temp - 1] = i - para_macro + 1; // 第i行对角线的值的位置
		a[num + temp - 1] = 0;
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 主对角线的初始值
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num - 1] += -Tb[j].Conductivity;
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num - 1] += -Tb[j].Conductivity;
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num] += -Tb[j].Conductivity;
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
						a[num] += -Tb[j].Conductivity;
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	if (Flag_eigen == false)
	{
		for (size_t i = 0; i < op + mp + 1; i++)
		{
			ia[i] += -1;
		}

		for (size_t i = 0; i < ia[op + mp]; i++)
		{
			ja[i] += -1;
		}
	}
}

void PNMsolver::CSR2COO()
{
	// 将CSR格式转换为COO格式
	int k = 0;
	for (size_t i = 0; i < op + mp + 1; i++)
	{
		ia[i] += 1;
	}

	for (size_t i = 0; i < ia[op + mp] - 1; i++)
	{
		ja[i] += 1;
	}

	for (int i = 0; i < op + mp; i++)
	{
		for (int j = ia[i] - 1; j < ia[i + 1] - 1; j++)
		{
			irn[k] = i + 1;
			jcn[k] = ja[j];
			k++;
		}
	}
}

void PNMsolver::Function_DS(double pressure)
{
	Ds = (Ds_LIST[6] - Ds_LIST[0]) / (50e6 - 1e6) * (pressure - 1e6) + Ds_LIST[0];
};

double PNMsolver::micro_permeability(double pre)
{
	ko = 1095e-21;
	ofstream ap_micro("ap_OM_type1.txt", ios::app);
	double z = compre(pre);
	double rho_g = pre * 0.016 / (z * 8.314 * 400); // kg/m3
	double viscos = visco(pre, z, 400);				// pa.s
	double Knusen_number = viscos / pre * sqrt(pi * z * 8.314 * 400 / (2 * 0.016)) / (33e-9);
	double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
	double beta = 4;
	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	ap_micro << pre / 1e6 << "	" << (viscos * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * pre, 2) / rho_g + Slip * ko) / 1e-21 << "	" << Slip * ko / 1e-21 << endl;
}

double PNMsolver::compre(double pressure)
{
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

	if ((int)solutions[2] != -1e4)
	{
		std::sort(solutions, solutions + 3, std::greater<double>());
		return solutions[0];
	}
	else
	{
		return solutions[0];
	}
};

double PNMsolver::visco(double p, double z, double T)
{
	p = 0.00014504 * p;														 // pa -> psi
	T = 1.8 * T;															 // k -> Rankin
	double density_of_gas = 28.967 * 0.5537 * p / (z * 10.732 * T) / 62.428; // g/cm3
	double Mg = 28.967 * 0.5537;
	double X = 3.448 + 986.4 / (T) + 0.001 * Mg; // T in R, M in g/mol
	double Y = 2.447 - 0.2224 * X;
	double K = (9.379 + 0.02 * Mg) * pow(T, 1.5) / (209.2 + 19.26 * Mg + T);
	return 1e-7 * K * exp(X * pow(density_of_gas, Y)); // cp -> Pa s
};

void PNMsolver::memory()
{
	std::vector<std::string> fileList = getFilesInFolder(folderPath);
	bool flag{false};
	for (const auto &file : fileList)
	{
		if (file.find(string("voxels_number")) != string::npos)
		{
			ifstream files(file, ios::in);
			if (files.is_open())
			{
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
			while ((idx = sline.find(shead, idx)) != string::npos && (idx1 = sline.find(sshead, idx1)) != string::npos)
			{
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
			while ((idx = sline.find(shead, idx)) != string::npos && (idx1 = sline.find("\t", idx1)) != string::npos)
			{
				istringstream ss(sline.substr(idx + 1, idx1 - idx - 1));
				int ii;
				ss >> ii;
				iputings.push_back(ii);
				idx++;
				idx1++;
			}

			while ((idx = sline.find(shead, idx)) != string::npos)
			{
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

	if (flag == false)
	{
		cout << "voxel file missed!" << endl;
		abort();
	}

	cout << "pn = " << pn << endl;
	cout << "tn = " << tn << endl;
	cout << "inlet = " << inlet << "; " << "outlet = " << outlet << "; " << "m_inlet = " << m_inlet << "; " << "m_outlet = " << m_outlet << "; " << "op = " << op << "; " << "mp = " << mp << "; " << endl;

	Pb = new pore[pn];
	Tb_in = new throat[2 * tn];
	Tb = new throatmerge[2 * tn];
}

void PNMsolver::initial_condition()
{
	double start = omp_get_wtime();

	for (int i = 0; i < pn; i++)
	{
		Pb[i].pressure = inlet_pre; //- double(double(i) / double(pn) * 100)
		Pb[i].pressure_old = Pb[i].pressure;
	}
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		Pb[i].pressure = outlet_pre;
		Pb[i].pressure_old = outlet_pre;
	}
	for (int i = pn - m_outlet; i < pn; i++)
	{
		Pb[i].pressure = outlet_pre;
		Pb[i].pressure_old = outlet_pre;
	}

	for (int i = 0; i < pn; i++)
	{
		Pb[i].mole_frac_co2 = 0.9; //- double(double(i) / double(pn) * 100)
		Pb[i].mole_frac_co2_old = Pb[i].mole_frac_co2;
	}
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		Pb[i].mole_frac_co2 = 0.1;
		Pb[i].mole_frac_co2_old = 0.1;
	}
	for (int i = pn - m_outlet; i < pn; i++)
	{
		Pb[i].mole_frac_co2 = 0.1;
		Pb[i].mole_frac_co2_old = 0.1;
	}
	double end = omp_get_wtime();
	printf("initial_condition start = %.16g\tend = %.16g\tdiff = %.16g\n",
		   start, end, end - start);
};

void PNMsolver::initial_condition(int flag)
{
	string filename = "Gas_production_";
	filename.append(to_string(Time_step));
	ifstream file(filename + ".vtk", ios::in);
	assert(file.is_open());
	string s;
	string head = "LOOKUP_TABLE table3";
	while (getline(file, s))
	{
		if (s.find(head) == 0)
		{
			break;
		}
	}

	for (int i = 0; i < pn; i++)
	{
		file >> Pb[i].pressure;
		Pb[i].pressure += outlet_pre;
		Pb[i].pressure_old = Pb[i].pressure;
	}
	file.close();
}

void PNMsolver::Paramentinput()
{
	cout << "亚分辨区域均质" << endl;
	std::vector<std::string> fileList = getFilesInFolder(folderPath);
	bool flag{false};
	for (const auto &file : fileList)
	{
		if (file.find(string("full_pore")) != string::npos)
		{
			ifstream porefile(file, ios::in);
			if (porefile.is_open())
			{
				flag = true;
			}

			for (int i = 0; i < pn; i++)
			{
				double waste{0};
				porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >> Pb[i].porosity >> Pb[i].km;
				Pb[i].km = ko;
				Pb[i].full_coord_ori = Pb[i].full_coord;
				Pb[i].full_accum_ori = Pb[i].full_accum;
			}
			porefile.close();
		}
	}
	if (flag == false)
	{
		cout << "porebody file missed!" << endl;
		abort();
	}

	flag = false;
	for (const auto &file : fileList)
	{
		if (file.find(string("full_throat")) != string::npos)
		{
			ifstream throatfile(file, ios::in);
			if (throatfile.is_open())
			{
				flag = true;
			}

			for (int i = 0; i < 2 * tn; i++)
			{
				throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
			}
			throatfile.close();
		}
	}
	if (flag == false)
	{
		cout << "throat file missed!" << endl;
		abort();
	}

	for (int i = 0; i < pn; i++)
	{
		Pb[i].X = voxel_size * Pb[i].X;
		Pb[i].Y = voxel_size * Pb[i].Y;
		Pb[i].Z = voxel_size * Pb[i].Z;
		Pb[i].Radiu = voxel_size * Pb[i].Radiu;
	}

	for (int i = 0; i < 2 * tn; i++)
	{
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu; // pnm部分为喉道的半径
		}
		else
		{
			Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu; // Darcy区的为接触面积
		}
		Tb_in[i].Length = voxel_size * Tb_in[i].Length;
		Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
		Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
		Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
	}
}

void PNMsolver::Paramentinput(int i) // 非均质微孔区域
{
	cout << "亚分辨区域非均质" << endl;
	std::vector<std::string> fileList = getFilesInFolder(folderPath);
	bool flag{false};
	for (const auto &file : fileList)
	{
		if (file.find(string("full_pore")) != string::npos)
		{
			ifstream porefile(file, ios::in);
			if (porefile.is_open())
			{
				flag = true;
			}
			for (int i = 0; i < pn; i++)
			{
				double waste{0};
				porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >> Pb[i].porosity >> Pb[i].km;
			}
			porefile.close();
		}
	}
	if (flag == false)
	{
		cout << "porebody file missed!" << endl;
		abort();
	}

	flag = false;
	for (const auto &file : fileList)
	{
		if (file.find(string("full_throat")) != string::npos)
		{
			ifstream throatfile(file, ios::in);
			if (throatfile.is_open())
			{
				flag = true;
			}

			for (int i = 0; i < 2 * tn; i++)
			{
				throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
			}
			throatfile.close();
		}
	}
	if (flag == false)
	{
		cout << "throat file missed!" << endl;
		abort();
	}

	for (int i = 0; i < pn; i++)
	{
		Pb[i].X = voxel_size * Pb[i].X;
		Pb[i].Y = voxel_size * Pb[i].Y;
		Pb[i].Z = voxel_size * Pb[i].Z;
		Pb[i].Radiu = voxel_size * Pb[i].Radiu;
	}

	for (int i = 0; i < 2 * tn; i++)
	{
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu; // pnm部分为喉道的半径
		}
		else
		{
			Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu; // Darcy区的为接触面积
		}
		Tb_in[i].Length = voxel_size * Tb_in[i].Length;
		Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
		Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
		Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
	}
}

void PNMsolver::para_cal()
{
	// 计算孔隙的体积
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type == 0)
		{
			Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3; // 孔隙网络单元
		}
		else if (Pb[i].type == 1)
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3); // 正方形微孔单元
		}
		else
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3) / 2; // 2×2×1、1×2×2和2×1×2的微孔网格
		}
	}

	// 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
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
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
		}
		else
		{
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
	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	Pb[0].full_accum = Pb[0].full_coord;

	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}

	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);

	coolist.resize(op + mp);  // 非进出口全配位数
	coolist3.resize(op + mp); // 非进出口的局部指标
	coolist4.resize(op + mp); // 非进出口的全局指标
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		int counter{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else
			{
				counter++;
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			int counter{0};
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else
			{
				counter++;
			}
		}
	}

	coolist2.resize(op + mp); // 非进出口累计全配位数
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < coolist2.size(); i++)
	{
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

void PNMsolver::para_cal_in_newton()
{
	// 计算压缩系数
	for (int i = 0; i < pn; i++)
	{
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
		// 计算克努森数
		double Knusen_number{0};
		double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure) / 2;
		double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
		double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (Tb_in[i].Radiu * 2);
		}
		else
		{
			Knusen_number = Average_visco / Average_pressure * sqrt(pi * Average_compre * 8.314 * Temperature / (2 * 0.016)) / (micro_radius * 2);
		}
		// 计算滑移项
		double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
		double beta = 4;
		double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
		Tb_in[i].Knusen = Knusen_number;
		Tb_in[i].Slip = Slip;
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
			{
				Tb_in[i].Length = 0.5 * voxel_size;
			}
			Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
		{
			temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
			}
			temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
			temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
			temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		}
		else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
		{
			temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
			}
			temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

			temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
			temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = temp11 * temp22 / (temp11 + temp22);
		}
		else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
		{
			Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = Tb_in[i].Radiu * Ds * n_max_ad / Tb_in[i].Length;
		}
		else
		{
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
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

void PNMsolver::para_cal(double mode)
{
	// 计算孔隙的体积
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type == 0)
		{
			Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3; // 孔隙网络单元
		}
		else if (Pb[i].type == 1)
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3); // 正方形微孔单元
		}
		else
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3) / 2; // 2×2×1、1×2×2和2×1×2的微孔网格
		}
	}

	// 计算压缩系数 气体粘度
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
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
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) reduction(+ : total_micro_free, total_micro_ad)
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
		total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
	}

	total_p = total_macro + total_micro_free + total_micro_ad;
	cout << "total_macro = " << total_macro << endl;
	cout << "total_micro_free = " << total_micro_free << endl;
	cout << "total_micro_ad = " << total_micro_ad << endl;
	cout << "total_p = " << total_p << endl;

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
																										   // 计算克努森数
	double Knusen_number{0};
	double Average_pressure{0};
	double Average_compre{0};
	double Average_visco{0};
	double Slip{0};
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA)) private(Knusen_number, Average_pressure, Average_compre, Average_visco, Slip, \
																temp1, temp2, temp11, temp22, angle1, angle2, length1, length2)
#endif
	for (int i = 0; i < 2 * tn; i++)
	{
		Knusen_number = 0;
		Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure) / 2;
		Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
		Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;
		Slip = 1;
		Tb_in[i].Knusen = Knusen_number;
		Tb_in[i].Slip = 1;
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
			{
				Tb_in[i].Length = 0.5 * voxel_size;
			}
			Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
		{
			temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
			}
			temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
			temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
			temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
		{
			temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
			}
			temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

			temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
			temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
		{
			Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else
		{
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
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
	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	Pb[0].full_accum = Pb[0].full_coord;

	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}

	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);

	coolist.resize(op + mp);
	coolist3.resize(op + mp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		int counter{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - inlet] += 1;
				coolist3[i - inlet].push_back(counter);
				counter++;
			}
			else
			{
				counter++;
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			int counter{0};
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp))
			{
				coolist[i - para_macro] += 1;
				coolist3[i - para_macro].push_back(counter);
				counter++;
			}
			else
			{
			}
		}
	}

	coolist2.resize(op + mp);
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < coolist2.size(); i++)
	{
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

void PNMsolver::para_cal_in_newton(double mode)
{
	// 计算压缩系数
	for (int i = 0; i < pn; i++)
	{
		Pb[i].compre = compre(Pb[i].pressure);
		Pb[i].visco = visco(Pb[i].pressure, Pb[i].compre, Temperature);
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
	for (int i = 0; i < 2 * tn; i++)
	{
		// 计算克努森数
		double Knusen_number{0};
		double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure) / 2;
		double Average_compre = (Pb[Tb_in[i].ID_1].compre + Pb[Tb_in[i].ID_2].compre) / 2;
		double Average_visco = (Pb[Tb_in[i].ID_1].visco + Pb[Tb_in[i].ID_2].visco) / 2;

		double Slip = 1;
		Tb_in[i].Knusen = Knusen_number;
		Tb_in[i].Slip = 1;
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
			{
				Tb_in[i].Length = 0.5 * voxel_size;
			}
			Tb_in[i].Conductivity = Slip * pi * pow(Tb_in[i].Radiu, 4) / (8 * Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
		{
			temp1 = Slip * pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * Average_visco);
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
			}
			temp2 = abs(Slip * Pb[Tb_in[i].ID_2].km * Tb_in[i].Radiu * angle2 / (Average_visco * length2));
			temp11 = abs(pi * Pb[Tb_in[i].ID_1].Radiu * Ds * n_max_ad);
			temp22 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle2 / length2);

			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
		{
			temp2 = Slip * pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * Average_visco);
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
			}
			temp1 = abs(Slip * Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (Average_visco * length1));

			temp11 = abs(Tb_in[i].Radiu * Ds * n_max_ad * angle1 / length1);
			temp22 = abs(pi * Pb[Tb_in[i].ID_2].Radiu * Ds * n_max_ad);
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
		{
			Tb_in[i].Conductivity = Slip * ko * Tb_in[i].Radiu / (Average_visco * Tb_in[i].Length);
			Tb_in[i].Surface_diff_conduc = 0;
		}
		else
		{
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
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
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
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
	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}
#pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;
#pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
}

void PNMsolver::PressureMatrix()
{
	/* -------------------------------------------------------------------------------------  */
	/* BULK PHASE EQUATION SOLEVR */
	/* -------------------------------------------------------------------------------------  */

	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	int counter = 0;
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < NA; i++)
	{
		COO_A[i].col = 0;
		COO_A[i].row = 0;
		COO_A[i].val = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < (op + mp); i++)
	{
		B[i] = 0;
	}

	// #ifdef _OPENMP
	// #pragma omp parallel for num_threads(int(OMP_PARA))
	// #endif
	for (int i = inlet; i < op + inlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
		}

		F = func_BULK_PHASE_FLOW_in_macro_produc(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);
		B[i - inlet] = -F.val();
		COO_A[i - inlet].row = i - inlet;
		COO_A[i - inlet].col = i - inlet;
		COO_A[i - inlet].val = Pi.d(0);

		size_t counter{0};	// 跳过进出口
		size_t counter1{0}; // COOA内存指标
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) // 连接的是大孔
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) // 连接的是微孔
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

	/* -------------------------------------------------------------------------------------  */
	/* 微孔组装 */
	/* -------------------------------------------------------------------------------------  */
	// micropore
	// #ifdef _OPENMP
	// #pragma omp parallel for num_threads(int(OMP_PARA))
	// #endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		reverse_mode<double> Pi, Wi, F;
		reverse_mode<double> *Pjs, *Wjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
		}

		F = func_BULK_PHASE_FLOW_in_micro_produc(Pi, Pjs, Wi, Wjs, i);
		F.diff(0, 1);

		B[i - para_macro] = -F.val();

		COO_A[i - para_macro].row = i - para_macro;
		COO_A[i - para_macro].col = i - para_macro;
		COO_A[i - para_macro].val = Pi.d(0);
		size_t counter{0};	// 跳过进出口
		size_t counter1{0}; // COOA内存指标
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) // 连接的是大孔
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) // 连接的是微孔
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

	double end = omp_get_wtime();
	printf("matrix diff = %.16g\n",
		   end - start);
	// ofstream B_OUT("B_OUT_ad.txt");

	// for (size_t i = 0; i < (op + mp); i++)
	// {
	// 	B_OUT << B[i] << endl;
	// }
}

void PNMsolver::Eigen_subroutine(Eigen::SparseMatrix<double, Eigen::RowMajor> &A0, Eigen::VectorXd &B0)
{
	using namespace Eigen;
	auto start = high_resolution_clock::now();

	for (int i = 0; i < (op + mp) * 2; ++i)
	{
		for (int j = ia[i]; j < ia[i + 1]; j++)
		{
			A0.coeffRef(i, ja[j - 1] - 1) = a[j - 1];
			// cout << "i = " << i << "	ja[j - 1] - 1 =	" << ja[j - 1] - 1 << "	a[j -1] = " << a[j - 1] << endl;
		}
	}

	for (int i = 0; i < (op + mp) * 2; i++)
	{
		B0[i] = B[i];
		/*cout << B0[i] << endl;*/
	}

	SimplicialLLT<SparseMatrix<double, RowMajor>> solver;
	// solver.setTolerance(pow(10, -5));
	// solver.setMaxIterations(1000);
	/*solver.setMaxIterations(3);*/
	// 计算分解
	solver.compute(A0);

	VectorXd x = solver.solve(B0);
	// std::cout << "#iterations:     " << solver.iterations() << std::endl;
	// iterations_number = solver.iterations();
	// std::cout << "estimated error: " << solver.error() << std::endl;

	/*for (int i = 0;i <op+mp;i++)
	{
		std::cout << "x["<<i<<"] = " << x[i]<<endl;
	}*/

	// 矩阵的无穷阶范数
	// norm_inf = x.lpNorm<Eigen::Infinity>();
	// 矩阵的二阶范数
	norm_inf = x.norm();

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += x[i - inlet];
		Pb[i].mole_frac_co2 += x[i - inlet + op + mp];

		// cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += x[i];
		Pb[i + inlet + outlet + m_inlet].mole_frac_co2 += x[i + op + mp];
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
	// endTime = clock();
	// cout<<"solver time:"<<(endTime-startTime)/1000<<endl;
}

void PNMsolver::Eigen_subroutine_per(Eigen::SparseMatrix<double, Eigen::RowMajor> &A0, Eigen::VectorXd &B0)
{
	using namespace Eigen;
	auto start = high_resolution_clock::now();

	for (int i = 0; i < op + mp; ++i)
	{
		for (int j = ia[i]; j < ia[i + 1]; j++)
		{
			A0.coeffRef(i, ja[j - 1] - 1) = a[j - 1];
			// cout << "i = " << i << "," << "j = " << ja[j - 1] - 1 << "," << "A = " << a[j - 1] << endl;
		}
	}

	for (int i = 0; i < op + mp; i++)
	{
		B0[i] = B[i];
		// cout << B0[i] << endl;
	}
	BiCGSTAB<SparseMatrix<double, RowMajor>> solver;
	solver.setMaxIterations(100000);
	solver.setTolerance(pow(10, -8));
	// 计算分解
	solver.compute(A0);
	VectorXd x = solver.solve(B0);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	iters_globa = solver.iterations();
	iterations_number = solver.iterations();
	std::cout << "estimated error: " << solver.error() << std::endl;

	/*for (int i = 0;i <op+mp;i++)
	{
		std::cout << "x["<<i<<"] = " << x[i]<<endl;
	}*/

	// 矩阵的无穷阶范数
	// norm_inf = x.lpNorm<Eigen::Infinity>();
	// 矩阵的二阶范数
	// norm_inf = x.norm();

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += x[i];
	}
	////outlet部分孔设置为解吸出口
	// for (int i = 0; i < inlet; i++)
	// {
	// 	Pb[i].pressure += x[Tb[i].ID_2 - inlet];
	// }
	// for (int i = macro_n; i < macro_n + m_inlet; i++)
	// {
	// 	Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
	// }
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Eigen subroutine Time-consuming = " << duration.count() / 1000 << "S" << endl;
	// endTime = clock();
	// cout<<"solver time:"<<(endTime-startTime)/1000<<endl;
}

double PNMsolver::Nor_inf(double A[])
{
	double temp1;
	temp1 = abs(A[0]);
	for (int i = 1; i < op + mp; i++)
	{
		if (temp1 < abs(A[i]))
		{
			temp1 = abs(A[i]);
		}
	}
	return temp1;
}

double PNMsolver::macro_outlet_flow()
{
	double Q_outlet = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double rho{0};
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				rho = Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
			}
			Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho; // 质量流量 g
		}
	}
	return abs(Q_outlet);
}

double PNMsolver::micro_outlet_free_flow()
{
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double rho{0};
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				rho = Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
			}
			Q_outlet += dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity * 16 * rho; // 质量流量 g
		}
	}
	return abs(Q_outlet);
}

double PNMsolver::micro_outlet_ad_flow()
{
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			Q_outlet += dt * Tb[j].Surface_diff_conduc * (K_langmuir * Pb[Tb[j].ID_1].pressure / (1 + K_langmuir * Pb[Tb[j].ID_1].pressure) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure)) * 1000;
		}
	}
	return abs(Q_outlet);
}

double PNMsolver::macro_outlet_Q()
{
	// ofstream out_flux("out_flux.txt");
	double Q_outlet = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			Q_outlet += (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity; // 体积流量
																								  // out_flux << abs(Q_outlet) << ";" << 1 << ";" << i << ";" << Pb[Tb[j].ID_1].pressure << ";" << Pb[Tb[j].ID_2].pressure << ";" << Tb[j].Conductivity << ";" << Pb[Tb[j].ID_1].Radiu << ";" << Pb[Tb[j].ID_2].Radiu << ";" << Tb[j].Radiu << ";" << Tb_in[j].Length << endl;
		}
	}
	return abs(Q_outlet);
}; // 出口大孔流量

double PNMsolver::micro_outlet_advec_Q()
{
	ofstream out_flux("out_flux.txt", ios::app);
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			Q_outlet += (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity; // 体积流量
			out_flux << abs(Q_outlet) << ";" << 0 << ";" << i << ";" << Pb[Tb[j].ID_1].pressure << ";" << Pb[Tb[j].ID_2].pressure << ";" << Tb[j].Conductivity << ";" << Pb[Tb[j].ID_1].Radiu << ";" << Pb[Tb[j].ID_2].Radiu << ";" << Tb[j].Radiu << ";" << Tb_in[j].Length << endl;
		}
	}
	return abs(Q_outlet);
}; // 出口微孔流量

double PNMsolver::micro_outlet_diff_Q()
{
	// ofstream out_flux("out_flux.txt",ios::app);
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double average_density = (Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) + Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature)) / 2;
			Q_outlet += Tb[j].Surface_diff_conduc * (K_langmuir * Pb[Tb[j].ID_1].pressure / (1 + K_langmuir * Pb[Tb[j].ID_1].pressure) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure)) / (average_density * 16e-3); // 体积流量
		}
		// out_flux << abs(Q_outlet) << endl;
	}
	return abs(Q_outlet);
}; // 出口吸附量

double PNMsolver::macro_mass_loss()
{
	double macro_mass_loss = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		macro_mass_loss += Pb[i].pressure_old * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
	}
	return (macro_mass_loss);
}

double PNMsolver::micro_free_mass_loss()
{
	double micro_mass_loss = 0;
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));			// kg/m3
		double n_ad_old = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)); // kg/m3

		micro_mass_loss += (porosity - n_ad_old / Rho_ad) * Pb[i].pressure_old * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
	}
	return (micro_mass_loss);
}

double PNMsolver::micro_ad_mass_loss() // - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature)
{
	double micro_ad_mass_loss = 0;
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
		double n_ad_old = n_max_ad * (K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old));

		micro_ad_mass_loss += (n_ad_old - n_ad_new) * 1000 * Pb[i].volume; // 微孔累计产气质量 单位g
	}
	return (micro_ad_mass_loss);
}

void PNMsolver::output(int n, bool m)
{
	ostringstream name;
	name << "Gas_production_" << int(n + 1) << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].Radiu << "\t";
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// 输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table3" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure - outlet_pre << endl;
	}
	// 输出进口
	outfile << "SCALARS inlet double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < inlet)
		{
			outfile << 0 << endl;
		}
		else if ((inlet + outlet + op <= i) && (i < inlet + outlet + op + m_inlet))
		{
			outfile << 1 << endl;
		}
		else if ((inlet + op <= i) && (i < inlet + outlet + op))
		{
			outfile << 2 << endl;
		}
		else if ((inlet + outlet + op + m_inlet + mp <= i) && (i < inlet + outlet + op + m_inlet + mp + m_outlet))
		{
			outfile << 3 << endl;
		}
		else
		{
			outfile << 4 << endl;
		}
	}

	// 输出进口
	outfile << "SCALARS type double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}
	outfile.close();
}

void PNMsolver::output(int n, int m)
{
	// ofstream main_path_macro("main_path_macro.txt");
	// ofstream main_path_micro("main_path_micro.txt");
	// ofstream main_path_double("main_path_double.txt");

	// ofstream sub_path_macro("sub_path_macro.txt");
	// ofstream sub_path_micro("sub_path_micro.txt");
	// ofstream sub_path_double("sub_path_double.txt");
	// ofstream flux("flux.txt");
	vector<double> macro_fluxes;
	vector<double> micro_fluxes;

	ostringstream name;
	name << "Permeability";
	name << to_string(inlet_pre);
	name << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// 输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table3" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure - outlet_pre << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}

	// 输出吼道信息
	outfile << "CELL_DATA"
			<< "\t" << Pb[pn - 1].full_accum << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table10" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << Tb[i].Radiu << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS Knusen double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << Tb[i].Knusen << endl;
	}

	outfile << "SCALARS neinuo double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		// 计算克努森数
		double neinuo{0};
		double Average_pressure = (Pb[Tb[i].ID_1].pressure + Pb[Tb[i].ID_2].pressure) / 2;
		double Average_compre = (Pb[Tb[i].ID_1].compre + Pb[Tb[i].ID_2].compre) / 2;
		double Average_visco = (Pb[Tb[i].ID_1].visco + Pb[Tb[i].ID_2].visco) / 2;
		if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n)
		{
			neinuo = Average_pressure * 0.016 / (Average_compre * 8.314 * Temperature) * Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) / (pi * pow(Tb[i].Radiu, 2)) * Tb[i].Radiu / Average_visco;
		}
		else
		{
			neinuo = Average_pressure * 0.016 / (Average_compre * 8.314 * Temperature) * Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) / (Tb[i].Radiu) / Average_visco;
		}
		// 计算滑移项
		outfile << neinuo << endl;
	}

	// outfile << "SCALARS free_gas_flux double 1" << endl;
	// outfile << "LOOKUP_TABLE table12" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
	// 	// flux << kkk << endl;
	// 	outfile << kkk << endl;
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n) // 大孔 主通道
	// 	{
	// 		macro_fluxes.push_back(kkk);
	// 	}
	// 	else
	// 	{
	// 		micro_fluxes.push_back(kkk);
	// 	}
	// }

	// auto macro_ptr = max_element(macro_fluxes.begin(), macro_fluxes.end());
	// auto micro_ptr = max_element(micro_fluxes.begin(), micro_fluxes.end());
	// auto thred1 = *macro_ptr * 0.2;
	// auto thred2 = *micro_ptr * 0.2;

	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
	// 	/*输出流量分布*/
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk >= thred1) // 大孔 主通道
	// 	{
	// 		Tb[i].main_free = int(0);
	// 		// main_path_macro << i << ";" << kkk << endl;
	// 	}
	// 	else if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk < thred1) // 大孔 次通道
	// 	{
	// 		Tb[i].main_free = int(3);
	// 		// sub_path_macro << i << ";" << kkk << endl;
	// 	}
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n && kkk >= thred2) // 微孔主通道
	// 	{
	// 		Tb[i].main_free = int(1);
	// 		// main_path_micro << i << ";" << kkk << endl;
	// 	}
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n && kkk < thred2) // 微孔 次通道
	// 	{
	// 		Tb[i].main_free = int(4);
	// 		// sub_path_micro << i << ";" << kkk << endl;
	// 	}
	// 	else if (((Tb[i].ID_1 < macro_n && Tb[i].ID_2 >= macro_n) || (Tb[i].ID_1 > macro_n && Tb[i].ID_2 <= macro_n)) && kkk >= thred2) // 连接主
	// 	{
	// 		Tb[i].main_free = int(2);
	// 		// main_path_double << i << ";" << kkk << endl;
	// 	}
	// 	else if (((Tb[i].ID_1 < macro_n && Tb[i].ID_2 >= macro_n) || (Tb[i].ID_1 > macro_n && Tb[i].ID_2 <= macro_n)) && kkk < thred2) // 连接次
	// 	{
	// 		Tb[i].main_free = int(5);
	// 		// sub_path_double << i << ";" << kkk << endl;
	// 	}
	// }

	// vector<double> micro_diff_flux;
	// vector<double> macro_diff_flux;

	// outfile << "SCALARS Surface_diffusion_flux double 1" << endl;
	// outfile << "LOOKUP_TABLE table13" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double average_density = (Pb[Tb[i].ID_1].pressure / (Pb[Tb[i].ID_1].compre * 8.314 * Temperature) + Pb[Tb[i].ID_2].pressure / (Pb[Tb[i].ID_2].compre * 8.314 * Temperature)) / 2;
	// 	double kkk = Tb[i].Surface_diff_conduc * abs(K_langmuir * Pb[Tb[i].ID_1].pressure / (1 + K_langmuir * Pb[Tb[i].ID_1].pressure) - K_langmuir * Pb[Tb[i].ID_2].pressure / (1 + K_langmuir * Pb[Tb[i].ID_2].pressure)) / (average_density * 16e-3);
	// 	outfile << kkk << endl;
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n) // 大孔 主通道
	// 	{
	// 		macro_diff_flux.push_back(kkk);
	// 	}
	// 	else
	// 	{
	// 		micro_diff_flux.push_back(kkk);
	// 	}
	// }
	// auto macro_diff_ptr = max_element(macro_diff_flux.begin(), macro_diff_flux.end());
	// auto diff_ptr = max_element(micro_diff_flux.begin(), micro_diff_flux.end());
	// auto Thred3 = *macro_diff_ptr * 0.2;
	// auto Thred4 = *diff_ptr * 0.2;
	// flux << "macro_max: " << *macro_ptr << endl
	// 	 << "micro_max: " << *micro_ptr << endl;
	// flux << "diff_max: " << *diff_ptr << endl;
	/*输出流量分布*/
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double average_density = (Pb[Tb[i].ID_1].pressure / (Pb[Tb[i].ID_1].compre * 8.314 * Temperature) + Pb[Tb[i].ID_2].pressure / (Pb[Tb[i].ID_2].compre * 8.314 * Temperature)) / 2;
	// 	double kkk = Tb[i].Surface_diff_conduc * abs(K_langmuir * Pb[Tb[i].ID_1].pressure / (1 + K_langmuir * Pb[Tb[i].ID_1].pressure) - K_langmuir * Pb[Tb[i].ID_2].pressure / (1 + K_langmuir * Pb[Tb[i].ID_2].pressure)) / (average_density * 16e-3);
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk > Thred3) // 大孔 主通道
	// 	{
	// 		Tb[i].main_surface = int(0);
	// 	}
	// 	else if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk <= Thred3) // 大孔 次通道
	// 	{
	// 		Tb[i].main_surface = int(3);
	// 	}
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n && kkk >= Thred4) // 微孔主通道
	// 	{
	// 		Tb[i].main_surface = int(1);
	// 	}
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n && kkk < Thred4) // 微孔 次通道
	// 	{
	// 		Tb[i].main_surface = int(4);
	// 	}
	// 	else if (((Tb[i].ID_1 < macro_n && Tb[i].ID_2 >= macro_n) || (Tb[i].ID_1 > macro_n && Tb[i].ID_2 <= macro_n)) && kkk >= Thred4) // 连接主
	// 	{
	// 		Tb[i].main_surface = int(2);
	// 	}
	// 	else if (((Tb[i].ID_1 < macro_n && Tb[i].ID_2 >= macro_n) || (Tb[i].ID_1 > macro_n && Tb[i].ID_2 <= macro_n)) && kkk < Thred4) // 连接次
	// 	{
	// 		Tb[i].main_surface = int(5);
	// 	}
	// }

	// outfile << "SCALARS main_free int 1" << endl;
	// outfile << "LOOKUP_TABLE table17" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	outfile << Tb[i].main_free << endl;
	// }
	// outfile << "SCALARS main_surface int 1" << endl;
	// outfile << "LOOKUP_TABLE table18" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	outfile << Tb[i].main_surface << endl;
	// }

	outfile.close();
	// main_path_macro.close();
	// main_path_micro.close();
	// main_path_double.close();
	// sub_path_macro.close();
	// sub_path_micro.close();
	// sub_path_double.close();
}

void PNMsolver::output(double n, double m)
{
	ofstream outfile("alone.vtk");
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS alone double 1" << endl;
	outfile << "LOOKUP_TABLE table019" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].full_coord == 0)
		{
			outfile << 1 << endl;
		}
		else
		{
			outfile << 0 << endl;
		}
	}

	// 输出孔类型信息
	outfile << "SCALARS inlet_outlet double 1" << endl;
	outfile << "LOOKUP_TABLE table019" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < inlet || ((inlet + op + outlet) < i && i < (inlet + op + outlet + m_inlet)))
		{
			outfile << 1 << endl;
		}
		else if (((inlet + op) < i && i < (inlet + op + outlet)) || ((inlet + op + outlet + m_inlet + mp) < i && i < (inlet + op + outlet + m_inlet + mp + m_outlet)))
		{
			outfile << 2 << endl;
		}
		else
		{
			outfile << 0 << endl;
		}
	}
}

void PNMsolver::output(double i)
{
	ofstream outfile("presentation.vtk");
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// 输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table3" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure - outlet_pre << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}

	// 吸附气体含量 kg
	outfile << "SCALARS adsorped_gas_amount double 1" << endl;
	outfile << "LOOKUP_TABLE table5" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type != 0)
		{
			auto n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			auto n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));
			auto out{Pb[i].volume * (n_ad1 - n_ad2)};
			outfile << out << endl;
		}
		else
		{
			outfile << 0 << endl;
		}
	}

	// 自由气体含量
	outfile << "SCALARS free_gas_amount double 1" << endl;
	outfile << "LOOKUP_TABLE table6" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type != 0)
		{
			auto compre_1 = compre(inlet_pre);
			auto compre_2 = compre(outlet_pre);
			auto n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			auto n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));
			auto out = inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 0.016 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 0.016 / (compre_2 * 8.314 * Temperature);
			outfile << out << endl;
		}
		else
		{
			auto compre_1 = compre(inlet_pre);
			auto compre_2 = compre(outlet_pre);
			auto out = inlet_pre * Pb[i].volume * 0.016 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 0.016 / (compre_2 * 8.314 * Temperature);
			outfile << out << endl;
		}
	}

	// 微孔直径分布
	outfile << "SCALARS Diameter double 1" << endl;
	outfile << "LOOKUP_TABLE table7" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].radius_micro * 2 << endl;
	}

	// 微孔孔隙度分布
	outfile << "SCALARS Porosity double 1" << endl;
	outfile << "LOOKUP_TABLE table8" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].porosity << endl;
	}

	// 微孔渗透率分布
	outfile << "SCALARS Permeability double 1" << endl;
	outfile << "LOOKUP_TABLE table9" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].km << endl;
	}

	// 输出吼道信息
	outfile << "CELL_DATA"
			<< "\t" << Pb[pn - 1].full_accum << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table10" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << Tb[i].Radiu << endl;
	}

	// outfile << "SCALARS Tij double 1" << endl;
	// outfile << "LOOKUP_TABLE table11" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	outfile << Tb[i].Conductivity << endl;
	// }

	// outfile << "SCALARS Tij_Bar double 1" << endl;
	// outfile << "LOOKUP_TABLE table17" << endl;
	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double average_rho = 0.016 * (Pb[Tb[i].ID_1].pressure / (Pb[Tb[i].ID_1].compre * 8.314 * Temperature) + Pb[Tb[i].ID_2].pressure / (Pb[Tb[i].ID_2].compre * 8.314 * Temperature)) / 2;
	// 	double average_pressure = (Pb[Tb[i].ID_1].pressure + Pb[Tb[i].ID_2].pressure) / 2;
	// 	outfile << Tb[i].Surface_diff_conduc * K_langmuir / (1 + K_langmuir * average_pressure * average_rho) << endl;
	// }
}

void PNMsolver::output(int n)
{
	ostringstream name;
	name << "Gas_production_" << int(n + 1) << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// 输出孔喉连接信息
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// 输出孔体信息
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// 输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// 输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table3" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure - outlet_pre << endl;
	}

	// 输出孔类型信息
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table4" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}

	// 输出吼道信息
	outfile << "CELL_DATA"
			<< "\t" << Pb[pn - 1].full_accum << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table10" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << Tb[i].Radiu << endl;
	}
	outfile.close();
}

void PNMsolver::Eigen_solver()
{
	Flag_eigen = true;
	auto start1 = high_resolution_clock::now();
	double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

	int n = 1;							   // label of output file
	int inter_n{0};						   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;				   // accumulation production
	ofstream outfile("Gasproduction.txt"); // output permeability;

	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	if (time_step != 0)
	{
		initial_condition(1);
		n = percentage_production_counter; // label of output file
	}
	else
	{
		initial_condition();
	}
	initial_condition();
	para_cal();
	PressureMatrix();

	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"
			<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
			<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
			<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
			<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

			<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

			<< "mass_conservation_error = " << 0 << "\t"
			<< "macro_loss = " << free_macro_loss << "\t"
			<< "free_micro_loss = " << free_micro_loss << "\t"
			<< "ad_micro_loss = " << ad_micro_loss << "\t"

			<< "acu_flow_macro = " << acu_flow_macro << "\t"
			<< "acu_free_micro = " << acu_free_micro << "\t"
			<< "acu_ad_micro = " << acu_ad_micro << "\t"
			<< "total_flow / total_p = " << total_flow / total_p << "\t"
			<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
			<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
			<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
			<< endl;
	output(-1);
	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine(A0, B0);
	do
	{
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			PressureMatrix();
			Eigen_subroutine(A0, B0);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step << endl;
			cout << endl;
		} while (norm_inf > eps);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			acu_flow_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre);
			double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
			double n_ad_old = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			acu_free_micro += (porosity - n_ad_old / Rho_ad) * inlet_pre * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre) - K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)); // 微孔累计产气质量 单位g
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
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"
				<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
				<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
				<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
				<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

				<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

				<< "mass_conservation_error = " << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
				<< "macro_loss = " << free_macro_loss << "\t"
				<< "free_micro_loss = " << free_micro_loss << "\t"
				<< "ad_micro_loss = " << ad_micro_loss << "\t"

				<< "acu_flow_macro = " << acu_flow_macro << "\t"
				<< "acu_free_micro = " << acu_free_micro << "\t"
				<< "acu_ad_micro = " << acu_ad_micro << "\t"
				<< "total_flow / total_p = " << total_flow / total_p << "\t"
				<< "acu_flow_macro / total_sub-resolution poresp = " << acu_flow_macro / total_p << "\t"
				<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
				<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
				<< "eps = " << eps << "\t"
				<< endl;

		for (int i = 0; i < pn; i++)
		{
			Pb[i].pressure_old = Pb[i].pressure;
			Pb[i].compre_old = Pb[i].compre;
		}

		if (inter_n < 10)
		{
			dt = dt * 2;
		}
		if (total_flow / total_p > 0.1 * n)
		{
			output(time_step);
			n++;
		}
		time_step++;
	} while (total_flow / total_p < 0.99);
	output(time_step);
	// total_flow / total_p < 0.99
	// while (time_step<100);  while (Error() > 1e-8);
	outfile.close();
	auto stop1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(stop1 - start1);
	cout << "Time-consuming = " << duration1.count() << " MS" << endl;
	ofstream out("calculate time.txt");
	out << duration1.count();
	out.close();
}

void PNMsolver::Eigen_solver_per()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;											 // label of output file
	int inter_n{0};										 // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;								 // accumulation production
	ofstream outfile("Apparent_Permeability_Eigen.txt"); // output permeability;
	Flag_eigen = true;
	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	initial_condition();
	para_cal();
	Matrix_permeability();

	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine_per(A0, B0);
	for (size_t i = 2; i < 52; i++)
	{
		para_cal_in_newton();
		Matrix_permeability();
		Eigen_subroutine_per(A0, B0);
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_permeability();
			Eigen_subroutine_per(A0, B0);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t" << endl;
		} while (norm_inf > eps_per);

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();
		micro_diff = micro_outlet_diff_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< inlet_pre / 1e6 << "\t"
				<< duration2.count() / 1000 << "s" << "\t"
				<< iters_globa
				<< endl;
		inlet_pre = i * 1e6;
		outlet_pre = inlet_pre - 100;
		Function_DS(inlet_pre);
		initial_condition();
		output(1, 1);
	}
	outfile.close();
}

void PNMsolver::AMGXsolver_subroutine_co2_mehane(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx)
{
	auto start = high_resolution_clock::now();
	static int icount{0};
	if (icount == 0)
	{
		AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
		icount += 1;
	}
	else
	{
		AMGX_matrix_replace_coefficients(A_amgx, n_amgx, nnz_amgx, a, 0);
	}
	AMGX_solver_setup(solver, A_amgx);
	AMGX_vector_upload(b_amgx, n_amgx, 1, B);
	AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
	AMGX_vector_download(solution_amgx, dX);

	norm_inf = 0;
	for (size_t i = 0; i < (op + mp) * 2; i++)
	{
		norm_inf += dX[i] * dX[i];
	}
	norm_inf = sqrt(norm_inf);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场 浓度场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += dX[i - inlet];
		Pb[i].mole_frac_co2 += dX[i - inlet + op + mp];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
		Pb[i + inlet + outlet + m_inlet].mole_frac_co2 += dX[i + op + mp];
	}
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/

	/*-----------------------------边界条件---------------------------------*/
	for (int i = 0; i < inlet; i++)
	{
		// Pb[i].pressure += dX[Tb[i].ID_2 - inlet];
		// Pb[i].mole_frac_co2 = inlet_co2_mole_frac;
	}
	for (int i = macro_n; i < macro_n + m_inlet; i++)
	{
		// Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
		// Pb[i].mole_frac_co2 = inlet_co2_mole_frac;
	}

	for (size_t i = inlet + op; i < inlet + op + outlet; i++)
	{
		// Pb[i].mole_frac_co2 = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].mole_frac_co2;
	}
	for (size_t i = macro_n + m_inlet + mp; i < macro_n + m_inlet + mp + m_outlet; i++)
	{
		// Pb[i].mole_frac_co2 = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].mole_frac_co2;
	}

	/*-----------------------------边界条件---------------------------------*/
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
}

void PNMsolver::AMGXsolver_subroutine(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx)
{
	auto start = high_resolution_clock::now();
	static int icount{0};
	if (icount == 0)
	{
		AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
		icount += 1;
	}
	else
	{
		AMGX_matrix_replace_coefficients(A_amgx, n_amgx, nnz_amgx, a, 0);
	}
	AMGX_solver_setup(solver, A_amgx);
	AMGX_vector_upload(b_amgx, n_amgx, 1, B);
	AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
	AMGX_vector_download(solution_amgx, dX);

	norm_inf = 0;
	for (size_t i = 0; i < op + mp; i++)
	{
		norm_inf += dX[i] * dX[i];
	}
	norm_inf = sqrt(norm_inf);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += dX[i - inlet];
		// cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
	}
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/

	/*-----------------------------边界条件---------------------------------*/
	for (int i = 0; i < inlet; i++)
	{
		Pb[i].pressure += dX[Tb[i].ID_2 - inlet];
	}
	for (int i = macro_n; i < macro_n + m_inlet; i++)
	{
		Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
	}
	/*-----------------------------边界条件---------------------------------*/
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
};

void PNMsolver::AMGXsolver_subroutine_per(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx)
{
	auto start = high_resolution_clock::now();
	static int icount{0};
	if (icount == 0)
	{
		AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
		icount += 1;
	}
	else
	{
		AMGX_matrix_replace_coefficients(A_amgx, n_amgx, nnz_amgx, a, 0);
	}
	AMGX_solver_setup(solver, A_amgx);
	AMGX_vector_upload(b_amgx, n_amgx, 1, B);
	AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
	AMGX_vector_download(solution_amgx, dX);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = dX[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure = dX[i];
	}
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/

	/*-----------------------------边界条件---------------------------------*/
	// for (int i = 0; i < inlet; i++)
	// {
	// 	Pb[i].pressure = inlet_pre;
	// }
	// for (int i = macro_n; i < macro_n + m_inlet; i++)
	// {
	// 	Pb[i].pressure = inlet_pre;
	// }
	/*-----------------------------边界条件---------------------------------*/
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
};

void PNMsolver::AMGXsolver()
{
	Flag_eigen = false;
	auto start1 = high_resolution_clock::now();
	double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

	int n{1};
	int inter_n{0};									// The interation of outer loop of Newton-raphoon method
	double total_flow = 0;							// accumulation production
	ofstream outfile("Transient_ad.txt", ios::app); // output permeability;

	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	if (time_step != 0)
	{
		initial_condition(1);
		n = percentage_production_counter; // label of output file
	}
	else
	{
		initial_condition();
	}
	para_cal();
	PressureMatrix();
	Matrix_COO2CSR();
	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/1.json"); // 200

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

	int n_amgx = op + mp;
	int nnz_amgx = ia[op + mp];
	AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
	AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
	AMGX_pin_memory(a, nnz_amgx * sizeof(double));
	AMGX_pin_memory(B, sizeof(double) * n_amgx);
	AMGX_pin_memory(dX, sizeof(double) * n_amgx);

	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 + machine_time << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"
			<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
			<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
			<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
			<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"
			<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"
			<< "mass_conservation_error = " << 0 << "\t"
			<< "macro_loss = " << free_macro_loss << "\t"
			<< "free_micro_loss = " << free_micro_loss << "\t"
			<< "ad_micro_loss = " << ad_micro_loss << "\t"
			<< "acu_flow_macro = " << acu_flow_macro << "\t"
			<< "acu_free_micro = " << acu_free_micro << "\t"
			<< "acu_ad_micro = " << acu_ad_micro << "\t"
			<< "total_flow / total_p = " << total_flow / total_p << "\t"
			<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
			<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
			<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
			<< endl;
	output(time_step - 1); // 初始状态
	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	AMGXsolver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	do
	{
		inter_n = 0;
		do
		{
			PressureMatrix();
			Matrix_COO2CSR();
			AMGXsolver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps);

		para_cal_in_newton();
		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			acu_flow_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre);
			double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
			double n_ad_old = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			acu_free_micro += (porosity - n_ad_old / Rho_ad) * inlet_pre * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre) - K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)); // 微孔累计产气质量 单位g
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
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"
				<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
				<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
				<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
				<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

				<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

				<< "mass_conservation_error = " << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
				<< "macro_loss = " << free_macro_loss << "\t"
				<< "free_micro_loss = " << free_micro_loss << "\t"
				<< "ad_micro_loss = " << ad_micro_loss << "\t"

				<< "acu_flow_macro = " << acu_flow_macro << "\t"
				<< "acu_free_micro = " << acu_free_micro << "\t"
				<< "acu_ad_micro = " << acu_ad_micro << "\t"
				<< "total_flow / total_p = " << total_flow / total_p << "\t"
				<< "acu_flow_macro / total_sub-resolution poresp = " << acu_flow_macro / total_p << "\t"
				<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
				<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
				<< "eps = " << eps << "\t"
				<< endl;

		for (int i = 0; i < pn; i++)
		{
			Pb[i].pressure_old = Pb[i].pressure;
			Pb[i].compre_old = Pb[i].compre;
		}

		if (inter_n < 10)
		{
			dt = dt * 2;
		}

		if (total_flow / total_p > 0.1 * n)
		{
			n++;
		}
		output(time_step);

		time_step++;
	} while (total_flow / total_p < 0.99);
	output(time_step);

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

void PNMsolver::Matrix_permeability()
{
	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	int temp2 = 0;

	ia[0] = 1;
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}

	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}

	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = inlet; i < op + inlet; i++)
	{
		B[i - inlet] = 0;
		temp = 0, temp1 = 0;
		num = ia[i - inlet];
		// macropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet) // 进口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}

				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - inlet + 1] = num1 + 1; // 前i行累计的非零值数量，其中1为ia[0]的值

		ja[num + temp - 1] = i - inlet + 1; // 第i行对角线的值的位置
		a[num + temp - 1] = 0;				// 主对角线的初始值

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
					}
					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	/* -------------------------------------------------------------------------------------  */
	/* 微孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		B[i - para_macro] = 0;

		temp = 0, temp1 = 0;
		num = ia[i - para_macro];
		// micropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进出口边界
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * Pb[Tb[j].ID_1].pressure / (1 + K_langmuir * Pb[Tb[j].ID_1].pressure) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * Pb[Tb[j].ID_1].pressure / (1 + K_langmuir * Pb[Tb[j].ID_1].pressure) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_1].pressure / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * Pb[Tb[j].ID_2].pressure / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * Pb[Tb[j].ID_1].pressure / (1 + K_langmuir * Pb[Tb[j].ID_1].pressure) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));

				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - para_macro + 1] = num1 + 1;		 // 前i行累计的非零值数量，其中1为ia[0]的值
		ja[num + temp - 1] = i - para_macro + 1; // 第i行对角线的值的位置
		a[num + temp - 1] = 0;
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 主对角线的初始值
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2);
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2);
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * Pb[Tb[j].ID_2].pressure, 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * Pb[Tb[j].ID_2].pressure, 2);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else // 上三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{

						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * Pb[Tb[j].ID_2].pressure, 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * Pb[Tb[j].ID_1].pressure, 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * Pb[Tb[j].ID_2].pressure, 2);
					}

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
		}
	}

	if (Flag_eigen == false)
	{
		for (size_t i = 0; i < op + mp + 1; i++)
		{
			ia[i] += -1;
		}

		for (size_t i = 0; i < ia[op + mp]; i++)
		{
			ja[i] += -1;
		}
	}
};

void PNMsolver::AMGX_permeability_solver()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													  // label of output file
	int inter_n{0};												  // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;										  // accumulation production
	ofstream outfile("apparent_Permeability_amgx.txt", ios::app); // output permeability;

	Flag_eigen = false;
	Function_DS(inlet_pre);
	memory();
	Paramentinput();
	initial_condition();
	para_cal();
	Matrix_permeability();

	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/FGMRES_AGGREGATION.json"); // for 505050
	// AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC"); // for lager

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

	int n_amgx = op + mp;
	int nnz_amgx = ia[op + mp];
	AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
	AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
	AMGX_pin_memory(a, nnz_amgx * sizeof(double));
	AMGX_pin_memory(B, sizeof(double) * n_amgx);
	AMGX_pin_memory(dX, sizeof(double) * n_amgx);

	// end AMGX initialization

	// ************ begin AMGX solver ************
	AMGXsolver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	for (size_t i = 2; i < 52; i++)
	{
		para_cal_in_newton();
		Matrix_permeability();
		AMGXsolver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_permeability();
			AMGXsolver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t" << endl;
		} while (norm_inf > eps_per);

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();
		micro_diff = micro_outlet_diff_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< inlet_pre / 1e6 << "\t"
				<< (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1.42925e-20 << "\t"
				<< total_macro << "\t"
				<< total_micro_free << "\t"
				<< total_micro_ad << "\t"
				<< endl;
		// output(1, 1);
		inlet_pre = i * 1e6;
		outlet_pre = inlet_pre - 100;
		Function_DS(inlet_pre);
		initial_condition();
		total_macro = 0;
		total_micro_free = 0;
		total_micro_ad = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			double compre_2 = compre(outlet_pre);
			total_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			double compre_2 = compre(outlet_pre);

			double n_ad1 = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			double n_ad2 = n_max_ad * (K_langmuir * outlet_pre / (1 + K_langmuir * outlet_pre));

			total_micro_free += inlet_pre * (porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
			total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
		}
	}

	outfile.close();

	/***********************销毁AMGX***************************/
	AMGX_unpin_memory(ia);
	AMGX_unpin_memory(ja);
	AMGX_unpin_memory(a);
	AMGX_unpin_memory(B);
	AMGX_unpin_memory(dX);

	AMGX_solver_destroy(solver);
	AMGX_vector_destroy(b_amgx);
	AMGX_vector_destroy(solution_amgx);
	AMGX_matrix_destroy(A_amgx);
	AMGX_resources_destroy(rsrc);
	AMGX_config_destroy(config);
	AMGX_finalize();
	// ************ end AMGX solver ************
};

reverse_mode<double> PNMsolver::func(reverse_mode<double> &Pi, reverse_mode<double> *&Pjs, int Pore_id)
{
	reverse_mode<double> RETURN;
	size_t counter{0};
	for (int j = Pb[Pore_id].full_accum - Pb[Pore_id].full_coord; j < Pb[Pore_id].full_accum; j++)
	{
		if (Tb[j].ID_2 < inlet) // 大孔进口
		{
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			else
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			counter++;
		}
		else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 大孔出口
		{
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			else
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			counter++;
		}
		else if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进口边界
		{
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			else
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			counter++;
		}
		else if (Tb[j].ID_2 >= pn - m_outlet) // 微孔出口边界
		{
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			else
			{
				RETURN += Tb[j].Conductivity * (Pi - Pb[Tb[j].ID_2].pressure);
			}
			RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pb[Tb[j].ID_2].pressure / (1 + K_langmuir * Pb[Tb[j].ID_2].pressure));
			counter++;
		}
		else
		{
			if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
			{
				RETURN += Tb[j].Conductivity * (Pi - Pjs[counter]);
			}
			else
			{
				RETURN += Tb[j].Conductivity * (Pi - Pjs[counter]);
			}
			RETURN += Tb[j].Surface_diff_conduc * (K_langmuir * Pi / (1 + K_langmuir * Pi) - K_langmuir * Pjs[counter] / (1 + K_langmuir * Pjs[counter]));
			counter++;
		}
	}
	return RETURN;
}

bool my_compare(vector<double> a, vector<double> b)
{
	if (a[0] != b[0])
		return a[0] > b[0]; // 第一级比较
	else
	{
		if (a[1] != b[1])
			return a[1] > b[1]; // 如果第一级相同，比较第二级
		else
			return a[2] > b[2]; // 如果第二级仍相同，比较第三级
	}
}

void PNMsolver::Matrix_permeability(double mode)
{
	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	int counter = 0;
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < NA; i++)
	{
		COO_A[i].col = 0;
		COO_A[i].row = 0;
		COO_A[i].val = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = inlet; i < op + inlet; i++)
	{
		reverse_mode<double> Pi, F;
		reverse_mode<double> *Pjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];

		Pi = Pb[i].pressure;

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
		}
		F = func(Pi, Pjs, i);
		F.diff(0, 1);

		B[i - inlet] = -F.val();
		COO_A[i - inlet].row = i - inlet;
		COO_A[i - inlet].col = i - inlet;
		COO_A[i - inlet].val = Pi.d(0);

		size_t counter{0};
		size_t counter1{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))
			{
				COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
				COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

/* -------------------------------------------------------------------------------------  */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------  */
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		reverse_mode<double> Pi{Pb[i].pressure}, F;
		reverse_mode<double> *Pjs;
		Pjs = new reverse_mode<double>[Pb[i].full_coord];

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 找到pjs
		{
			Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure;
		}
		F = func(Pi, Pjs, i);
		F.diff(0, 1);

		B[i - para_macro] = -F.val();
		COO_A[i - para_macro].row = i - para_macro;
		COO_A[i - para_macro].col = i - para_macro;
		COO_A[i - para_macro].val = Pi.d(0);
		size_t counter{0};
		size_t counter1{0};
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))
			{
				COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - para_macro;
				COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);
				counter++;
				counter1++;
			}
			else
			{
				counter++;
			}
		}
	}

	double end = omp_get_wtime();
	printf("matrix diff = %.16g\n",
		   end - start);
}

void PNMsolver::AMGX_permeability_solver(double mode)
{
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;											 // label of output file
	double total_flow = 0;								 // accumulation production
	ofstream outfile("Intrinsic_permeability_amgx.txt"); // output permeability;

	Flag_eigen = false;
	memory();
	Paramentinput();
	initial_condition();
	para_cal(1);
	Matrix_permeability(1);
	Matrix_COO2CSR();

	double start = omp_get_wtime();
	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/FGMRES_AGGREGATION.json");

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

	int n_amgx = op + mp;
	int nnz_amgx = ia[op + mp];
	AMGX_pin_memory(ia, (n_amgx + 1) * sizeof(int));
	AMGX_pin_memory(ja, nnz_amgx * sizeof(int));
	AMGX_pin_memory(a, nnz_amgx * sizeof(double));
	AMGX_pin_memory(B, sizeof(double) * n_amgx);
	AMGX_pin_memory(dX, sizeof(double) * n_amgx);

	// end AMGX initialization

	// ************ begin AMGX solver ************
	AMGXsolver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	double end = omp_get_wtime();
	outfile << "solve time = " << end - start << "\t"
			<< "permeability = " << (macro + micro_advec) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << endl;
	// output(1, 1);
	outfile.close();

	/***********************销毁AMGX***************************/
	AMGX_unpin_memory(ia);
	AMGX_unpin_memory(ja);
	AMGX_unpin_memory(a);
	AMGX_unpin_memory(B);
	AMGX_unpin_memory(dX);

	AMGX_solver_destroy(solver);
	AMGX_vector_destroy(b_amgx);
	AMGX_vector_destroy(solution_amgx);
	AMGX_matrix_destroy(A_amgx);
	AMGX_resources_destroy(rsrc);
	AMGX_config_destroy(config);
	AMGX_finalize();
	// ************ end AMGX solver ************
};

void PNMsolver::mumps_solver(int MYID)
{
	Flag_eigen = false;
	auto start1 = high_resolution_clock::now();
	double acu_flow_macro{0}, acu_free_micro{0}, acu_ad_micro{0};

	int n = 1;						   // label of output file
	int inter_n{0};					   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;			   // accumulation production
	ofstream outfile("Transient.txt"); // output permeability;

	memory();
	Paramentinput();
	initial_condition();
	para_cal();
	PressureMatrix();
	CSR2COO();

	/* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
	DMUMPS_STRUC_C id;
	MUMPS_INT nn = op + mp;
	MUMPS_INT8 nnz = ia[op + mp] - 1;
	id.comm_fortran = USE_COMM_WORLD;
	id.par = 1;
	id.sym = 0;
	id.job = JOB_INIT;
	dmumps_c(&id);

	/* Define the problem on the host */
	if (MYID == 0)
	{
		id.n = nn;
		id.nnz = nnz;
		id.irn = irn;
		id.jcn = jcn;
		id.a = a;
		id.rhs = B;
	}
#define ICNTL(I) icntl[(I) - 1] /* macro s.t. indices match documentation */
	/* No outputs */
	id.ICNTL(1) = -1;
	id.ICNTL(2) = -1;
	id.ICNTL(3) = -1;
	id.ICNTL(4) = 0;

	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 << "\t"
			<< "physical_time = " << time_all << "\t"
			<< "dt = " << dt << "\t"
			<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
			<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
			<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
			<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

			<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

			<< "mass_conservation_error = " << 0 << "\t"
			<< "macro_loss = " << free_macro_loss << "\t"
			<< "free_micro_loss = " << free_micro_loss << "\t"
			<< "ad_micro_loss = " << ad_micro_loss << "\t"

			<< "acu_flow_macro = " << acu_flow_macro << "\t"
			<< "acu_free_micro = " << acu_free_micro << "\t"
			<< "acu_ad_micro = " << acu_ad_micro << "\t"
			<< "total_flow / total_p = " << total_flow / total_p << "\t"
			<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
			<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
			<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
			<< endl;
	output(-1);
	mumps_subroutine(id, MYID);
	do
	{
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			PressureMatrix();
			CSR2COO();
			mumps_subroutine(id, MYID);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step << endl;
			cout << endl;
		} while (norm_inf > eps);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre);
			acu_flow_macro += inlet_pre * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre);
			double n_ad_new = n_max_ad * (K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure));
			double n_ad_old = n_max_ad * (K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre));
			acu_free_micro += (porosity - n_ad_old / Rho_ad) * inlet_pre * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (porosity - n_ad_new / Rho_ad) * Pb[i].pressure * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * inlet_pre / (1 + K_langmuir * inlet_pre) - K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)); // 微孔累计产气质量 单位g
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
		outfile << "inner loop = " << inter_n << "\t"
				<< "machine_time = " << duration2.count() / 1000 << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"
				<< "Q_outlet_macro = " << Q_outlet_macro << "\t"
				<< "Q_outlet_free_micro = " << Q_outlet_free_micro << "\t"
				<< "Q_outlet_ad_micro = " << Q_outlet_ad_micro << "\t"
				<< "total_out_flow = " << Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro << "\t"

				<< "total_mass_loss = " << free_macro_loss + free_micro_loss + ad_micro_loss << "\t"

				<< "mass_conservation_error = " << abs(Q_outlet_macro + Q_outlet_free_micro + Q_outlet_ad_micro - abs(free_macro_loss + free_micro_loss + ad_micro_loss)) / (free_macro_loss + free_micro_loss + ad_micro_loss) << "\t"
				<< "macro_loss = " << free_macro_loss << "\t"
				<< "free_micro_loss = " << free_micro_loss << "\t"
				<< "ad_micro_loss = " << ad_micro_loss << "\t"

				<< "acu_flow_macro = " << acu_flow_macro << "\t"
				<< "acu_free_micro = " << acu_free_micro << "\t"
				<< "acu_ad_micro = " << acu_ad_micro << "\t"
				<< "total_flow / total_p = " << total_flow / total_p << "\t"
				<< "acu_flow_macro / total_p = " << acu_flow_macro / total_p << "\t"
				<< "acu_flow_micro / total_p = " << acu_free_micro / total_p << "\t"
				<< "acu_ad_micro / total_p = " << acu_ad_micro / total_p << "\t"
				<< endl;

		for (int i = 0; i < pn; i++)
		{
			Pb[i].pressure_old = Pb[i].pressure;
			Pb[i].compre_old = Pb[i].compre;
		}

		if (inter_n < 20)
		{
			dt = dt * 2;
		}
		// if (inter_n > 100)
		// {
		// 	dt = dt / 2;
		// }
		if (total_flow / total_p > 0.1 * n)
		{
			output(time_step);
			n++;
		}
		// if (total_flow / total_p > 0.95)
		// {
		// 	output(time_step);
		// }

		time_step++;
	} while (total_flow / total_p < 0.99);
	output(time_step);
	// total_flow / total_p < 0.99
	// while (time_step<100);  while (Error() > 1e-8);
	outfile.close();
	auto stop1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(stop1 - start1);
	cout << "Time-consuming = " << duration1.count() << " MS" << endl;
	ofstream out("calculate time.txt");
	out << duration1.count();
	out.close();

	/* Terminate instance. */
	id.job = JOB_END;
	dmumps_c(&id);
	if (MYID == 0)
	{
		if (!error)
		{
			printf("Nothing wrong");
		}
		else
		{
			printf("An error has occured, please check error code returned by MUMPS.\n");
		}
	}
	MPI_Finalize();
}

void PNMsolver::mumps_subroutine(DMUMPS_STRUC_C &id, int MYID)
{
	auto start = high_resolution_clock::now();
	static int count = 0;
	if (count == 0)
	{
	}
	else
	{
		if (MYID == 0)
		{
			id.irn = irn;
			id.jcn = jcn;
			id.a = a;
			id.rhs = B;
		}
		count = 1;
	}

	/* Call the MUMPS package (analyse, factorization and solve). */
	id.job = 6;
	dmumps_c(&id);
	if (id.infog[0] < 0)
	{
		printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
			   MYID, id.infog[0], id.infog[1]);
		error = 1;
	}

	for (size_t i = 0; i < op + mp; i++)
	{
		dX[i] = B[i];
	}

	norm_inf = 0;
	for (size_t i = 0; i < op + mp; i++)
	{
		norm_inf += dX[i] * dX[i];
	}
	norm_inf = sqrt(norm_inf);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += dX[i - inlet];
		// cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
	}
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/

	/*-----------------------------边界条件---------------------------------*/
	for (int i = 0; i < inlet; i++)
	{
		Pb[i].pressure += dX[Tb[i].ID_2 - inlet];
	}
	for (int i = macro_n; i < macro_n + m_inlet; i++)
	{
		Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
	}
	/*-----------------------------边界条件---------------------------------*/
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
};

void PNMsolver::mumps_subroutine_per(DMUMPS_STRUC_C &id, int MYID)
{
	auto start = high_resolution_clock::now();
	static int count = 0;
	if (count == 0)
	{
	}
	else
	{
		if (MYID == 0)
		{
			id.irn = irn;
			id.jcn = jcn;
			id.a = a;
			id.rhs = B;
		}
		count = 1;
	}

	omp_set_num_threads(20);
	/* Call the MUMPS package (analyse, factorization and solve). */
	id.job = 6;
	dmumps_c(&id);
	if (id.infog[0] < 0)
	{
		printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
			   MYID, id.infog[0], id.infog[1]);
		error = 1;
	}

	for (size_t i = 0; i < op + mp; i++)
	{
		dX[i] = B[i];
	}

	norm_inf = 0;
	for (size_t i = 0; i < op + mp; i++)
	{
		norm_inf += dX[i] * dX[i];
	}
	norm_inf = sqrt(norm_inf);

	/*--------------------------x(t+dt) = x(t) + dx----------------------*/
	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure += dX[i - inlet];
		// cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
	}
	/*--------------------------x(t+dt) = x(t) + dx----------------------*/

	/*-----------------------------边界条件---------------------------------*/
	// for (int i = 0; i < inlet; i++)
	// {
	// 	Pb[i].pressure += dX[Tb[i].ID_2 - inlet];
	// }
	// for (int i = macro_n; i < macro_n + m_inlet; i++)
	// {
	// 	Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
	// }
	/*-----------------------------边界条件---------------------------------*/
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time-consuming = " << duration.count() << " MS" << endl;
};

void PNMsolver::mumps_permeability_solver(int MYID, double kk)
{
	Flag_eigen = {false};
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;											  // label of output file
	int inter_n{0};										  // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;								  // accumulation production
	ofstream outfile("Intrinsic_Permeability_mumps.txt"); // output permeability;

	memory();
	Paramentinput();
	initial_condition();
	para_cal(1);
	Matrix_permeability(1);
	CSR2COO();
	/* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
	DMUMPS_STRUC_C id;
	MUMPS_INT nn = op + mp;
	MUMPS_INT8 nnz = ia[op + mp] - 1;
	id.comm_fortran = USE_COMM_WORLD;
	id.par = 1;
	id.sym = 2;
	id.job = JOB_INIT;
	dmumps_c(&id);

	/* Define the problem on the host */
	if (MYID == 0)
	{
		id.n = nn;
		id.nnz = nnz;
		id.irn = irn;
		id.jcn = jcn;
		id.a = a;
		id.rhs = B;
	}
#define ICNTL(I) icntl[(I) - 1] /* macro s.t. indices match documentation */
	/* No outputs */
	id.ICNTL(1) = 6;
	id.ICNTL(2) = 0;
	id.ICNTL(3) = 6;
	id.ICNTL(4) = 2;
	auto stop3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(stop3 - start1);
	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration3.count() / 1000 << "\t"
			<< endl;
	mumps_subroutine_per(id, MYID);
	// ************ begin Mumps solve ************
	do
	{
		para_cal_in_newton(1);
		Matrix_permeability(1);
		CSR2COO();
		mumps_subroutine_per(id, MYID);
		// MKLsolve();
		inter_n++;
		cout << "Inf_norm = " << norm_inf << "\t\t"
			 << "dt = " << dt << "\t\t"
			 << "inner loop = " << inter_n
			 << "\t\t"
			 << "outer loop = " << time_step << endl;
		cout << endl;
	} while (norm_inf > eps);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();
	micro_diff = micro_outlet_diff_Q();

	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start1);

	outfile << "inner loop = " << inter_n << "\t"
			<< "machine_time = " << duration2.count() / 1000 << "\t"
			<< "permeability = " << (macro + micro_advec + micro_diff) * visco(inlet_pre, compre(inlet_pre), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << endl;
	// output(1, 1);
	outfile.close();

	/* Terminate instance. */
	id.job = JOB_END;
	dmumps_c(&id);
	if (MYID == 0)
	{
		if (!error)
		{
			printf("Nothing wrong");
		}
		else
		{
			printf("An error has occured, please check error code returned by MUMPS.\n");
		}
	}
	MPI_Finalize();
};

void PNMsolver::Matrix_COO2CSR()
{
	// int num_rows = 2 * (op + mp);
	// int nnz = 4 * NA;
	int num_rows = op + mp;
	int nnz = NA;

	qsort(COO_A, nnz, sizeof(coo), sort_by_row); // sort by row

	ofstream COOA_OUT("COOA_ad_sorted.txt");

	for (size_t i = 0; i < nnz; i++)
	{
		COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val << endl;
	}

#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < num_rows + 1; i++)
	{
		ia[i] = 0;
	}

	for (int i = 0; i < nnz; i++)
	{
		ia[COO_A[i].row + 1]++;
		/*        printf("row:%d,column:%d,value:%f \n", idx_tmp[i].row, idx_tmp[i].col, idx_tmp[i].val);*/
	}
	// prefix-scan
	for (int i = 1; i <= num_rows; i++)
	{
		ia[i] = ia[i] + ia[i - 1];
		/*        printf("%d \n", rows_offsets[i]);*/
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < nnz; i++)
	{
		ja[i] = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (size_t i = 0; i < num_rows + 1; i++)
	{
		a[i] = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < nnz; i++)
	{
		ja[i] = COO_A[i].col;
		a[i] = COO_A[i].val;
	}
	double end = omp_get_wtime();
	printf("coo2csr diff = %.16g\n",
		   end - start);

	if (Flag_eigen == true)
	{
		for (size_t i = 0; i < num_rows + 1; i++)
		{
			ia[i] += 1;
		}

		for (size_t i = 0; i < ia[num_rows]; i++)
		{
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
};

int main(int argc, char **argv)
{
	// argc = 1;
	// char *name = "Project";
	// argv = &name;
	// int myid, ierr;
	// int error = 0;
	// ierr = MPI_Init(&argc, &argv);
	// ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	char *buf;
	buf = get_current_dir_name();
	folderPath.assign(buf);
	cout << folderPath << endl;

	PNMsolver Berea;
	/*产气模拟*/
	// Berea.Eigen_solver();
	// Berea.mumps_solver(myid);
	// Berea.AMGXsolver();
	// Berea.AMGXsplver_REV();
	/*渗透率计算*/
	// Berea.Eigen_solver_per();
	// Berea.mumps_permeability_solver(myid);
	// double start = omp_get_wtime();
	// Berea.AMGX_permeability_solver(1);
	// double end = omp_get_wtime();
	// cout << "total time = " << end - start << endl;
	// Berea.Intrinsic_permeability(myid); //各种直接法 解线性方程组
	// Berea.Mysolver_for_Appermeability();
	// Berea.Mysolver_for_inpermeability();

	/*二氧化碳驱替甲烷*/
	// double start = omp_get_wtime();
	// Berea.AMGX_CO2_methane_solver();
	Berea.eigen_CO2_methane_solver();
	// double end = omp_get_wtime();

	// for (size_t i = 1; i < 51; i++)
	// {
	// 	Berea.Function_DS(i * 1e6);
	// 	Berea.micro_permeability(i * 1e6);
	// }

	/*展示*/
	// Berea.memory();
	// Berea.Paramentinput(int(1)); //读取非均质
	// Berea.Paramentinput();
	// Berea.initial_condition();
	// Berea.para_cal();
	// Berea.output(double(1));
	// Berea.output(double(1), double(1)); // 输出单重孔网

	/*计算密度和粘度*/
	// ofstream gas_density_visco("density.txt");
	// for (size_t i = 0; i < 51; i++)
	// {
	// 	auto z{Berea.compre(i * 1e6)};
	// 	// gas_density_visco << i << ";" << i * 1e6 * 0.016 / (z * 8.314 * 400) << ";" << Berea.visco(i*1e6,z,400) << ";" << (porosity-n_max_ad*K_langmuir*i*1e6/(1+K_langmuir*i*1e6)/Rho_ad)*i * 1e6 * 0.016 / (z * 8.314 * 400) << endl;
	// 	gas_density_visco << K_langmuir*i*1e6/(1+K_langmuir * i *1e6)<< endl;
	// }
	// gas_density_visco.close();

	/*计算Tij*/
	// ofstream Tij("Tij.txt");
	// for (size_t i = 1; i < 51; i++)
	// {
	// 	auto pressure = i * 1e6;
	// 	auto R = 75e-9;
	// 	auto z{Berea.compre(pressure)};
	// 	auto vis(Berea.visco(pressure, z, Temperature));
	// 	double rho = 0.016 * pressure / (z * 8.314 * Temperature);
	// 	auto Knusen_number = vis / pressure * sqrt(pi * z * 8.314 * Temperature / (2 * 0.016)) / R;
	// 	double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
	// 	double beta = 4;
	// 	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	// 	auto out = Slip * pi * pow(R, 4) / (8 * vis * R);
	// 	Tij << "Tij=" << out << ";Knusen_number=" << Knusen_number << ";slip=" << Slip << ";pressure=" << pressure / 1e6 << endl;
	// }
	// Tij.close();

	// ofstream Tij_micro("Tij_micro.txt");
	// for (size_t i = 1; i < 51; i++)
	// {
	// 	auto pressure = i * 1e6;
	// 	auto R = 4e-9;
	// 	auto r = 1.8e-9;
	// 	auto km = 20e-21;
	// 	auto z{Berea.compre(pressure)};
	// 	auto vis(Berea.visco(pressure, z, Temperature));
	// 	double rho = 0.016 * pressure / (z * 8.314 * Temperature);
	// 	auto Knusen_number = vis / pressure * sqrt(pi * z * 8.314 * Temperature / (2 * 0.016)) / r;
	// 	double alpha = 1.358 * 2 / pi * atan(4 * pow(Knusen_number, 0.4));
	// 	double beta = 4;
	// 	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	// 	auto out = Slip * pi * pow(R, 2) * km / (vis * R);
	// 	Tij_micro << "Tij=" << out << ";Knusen_number=" << Knusen_number << ";slip=" << Slip << ";pressure=" << pressure / 1e6 << endl;
	// }
	// Tij_micro.close();

	// ofstream Tij_bar("Tij_bar.txt");
	// for (size_t i = 1; i < 51; i++)
	// {
	// 	auto pressure = i * 1e6;
	// 	Ds = (Ds_LIST[6] - Ds_LIST[0]) / (50e6 - 1e6) * (pressure - 1e6) + Ds_LIST[0];
	// 	auto R = 75e-9;
	// 	auto r = 1.8e-9;
	// 	auto length = 4e-9;
	// 	auto z{Berea.compre(pressure)};
	// 	auto vis(Berea.visco(pressure, z, Temperature));
	// 	double rho = 0.016 * pressure / (z * 8.314 * Temperature);
	// 	auto out = pi * pow(length, 2) * Ds * n_max_ad * K_langmuir / (1 + K_langmuir * pressure) / (rho * length);
	// 	Tij_bar << "Tij_bar=" << out << ";pressure=" << pressure / 1e6 << endl;
	// }
	// Tij_bar.close();

	// ofstream nad("nad.txt");
	// ofstream nn("nn.txt");
	// for (size_t i = 1; i < 51; i++)
	// {
	// 	auto pressure = i;
	// 	auto z{Berea.compre(pressure * 1e6)};
	// 	double rho = 0.016 * pressure * 1e6 / (z * 8.314 * Temperature);
	// 	auto out = 3.5 * pressure / (3.719 + pressure);
	// 	auto o = 44.5 * pressure / (25 + pressure);
	// 	nad << "pressure=" << pressure << ";gas=" << (porosity - out / Rho_ad) * rho << ";porosity=" << (porosity - out / Rho_ad) << ";rho = " << rho << endl;
	// 	nn << "pressure=" << pressure << ";gas=" << (porosity - o / Rho_ad) * rho << ";porosity=" << (porosity - o / Rho_ad) << ";rho = " << rho << endl;
	// }
	// nad.close();
	// nn.close();
	// free(buf);
	// cout << "计算结束" << endl;
	return 0;
}