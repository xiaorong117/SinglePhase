#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"
#include <omp.h>
#include <ctime>
#include <chrono>
#include <set>	   // STL set
#include <numeric> // accumulate

#include <sys/types.h>
#include <dirent.h>
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

#define OMP_PARA 20
using namespace std;
using namespace std::chrono;

// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
double iters_globa{0};
////常量设置
double pi = 3.1415927;
double gas_vis = 2e-5;	  // 粘度
double porosity = 0.1342; // 孔隙率   含水改成0.05，不含水0.1
double micro_radius{3.48e-9};
double ko = 12e-21;				// 微孔达西渗透率 m^2
double inlet_pre = 100;			// 进口压力 Pa
double outlet_pre = 0;			// 出口压力 Pa
double D = 9e-9;				// 扩散系数
double Effect_D = 0.05 * D;		// 微孔中的有效扩散系数
double voxel_size = 320e-9;		// 像素尺寸，单位m    5.345e-6 8e-9 320e-9 for REV
double domain_size_cubic = 384; // 384 for REV
double domain_length = 0;
double T_critical{190.564};		// 甲烷的临界温度 190.564K
double P_critical{4.599 * 1e6}; // 甲烷的临界压力 4.599MPa
double Temperature{400};		// 温度
double Rho_ad{400};				// kg/m3
double n_max_ad{44.8};			// kg/m3
double K_langmuir{4e-8};		// Pa^(-1)
double Ds{2.46e-8};				// m2/s

double porosity_OMHP1{0.243};	 // 含水时 porosity_1 会变
double porosity_OMHP2{0.243};	 // 含水时 porosity 2 不变
double porosity_OMLP1{0.081};	 // 含水时 porosity 1 会变
double porosity_OMLP2{0.081};	 // 含水时 porosity 2 不变
double porosity_clay_HP1{0.081}; // 含水时 porosity 1 改变
double porosity_clay_HP2{0.081}; // 含水时 porosity 2 不变
double porosity_clay_LP1{0.081}; // 含水时 porosity 1 改变
double porosity_clay_LP2{0.081}; // 含水时 porosity 2 不变
double micro_porosity_HP{porosity};
double micro_porosity_LP{porosity};
double micro_porosity_Clay_HP{porosity};
double micro_porosity_Clay_LP{porosity};

double swww_clay{0}; // 含水改成 1，不含水0
double swww_om{0};	 // 含水改成 0.5，不含水0
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
double K_Clay_HP{a_clay * pow(Sw_clay - Sw_max_clay, 6)};
double K_Clay_LP{0};

double refer_pressure{1e6};
double L_CLAY_HP{20e-9};
double L_CLAY_LP{20e-9};
double L_OM_HP{1000e-9};
double L_OM_LP{20e-9};

vector<double> Ds_LIST({8.32e-9, 9.52e-9, 1.14e-8, 1.44e-8, 1.77e-8, 2.10e-8, 2.46e-8});
int Time_step{0};
int percentage_production_counter{1};
double pyhsic_time{0};
double machine_time{0};

int Flag_eigen{true};
int Flag_intri_per{true};
int Flag_Hybrid{true};
int Flag_homo{true};
int Flag_outputvtk{true};
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

int Mode{0};

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
	int type{0};
	int main_gate{0};
	double pressure{0};
	double pressure_old{0};
	double volume{0};
	double compre{0};
	double compre_old{0};
	double visco{0};
	double visco_old{0};
	double km{0};
	double REV_k{0};
	double porosity{0};
	double REV_porosity1{0};
	double REV_porosity2{0};
	double REV_micro_porosity{0};
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
	// 申请孔喉的动态存储空间
	pore *Pb;
	throat *Tb_in;
	throatmerge *Tb;

	double error;
	int time_step = Time_step;
	double time_all = pyhsic_time;
	double dt = 1e-7;
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

	double total_clay_HP{0};
	double total_clay_LP{0};
	double total_fracture{0};
	double total_macropores{0};
	double total_OM_HP_free{0};
	double total_OM_LP_free{0};
	double total_OM_HP_ad{0};
	double total_OM_LP_ad{0};

	double norm_inf = 0;
	double eps = 1e-5;	   // set residual for dx
	double eps_per = 1e-3; // set residual for dx
	int force_out = 20;
	int double_time = 10;
	double maximum_dt = 1e-1;

	int iterations_number = 0;
	// double total_p=2.75554e-8;

	void memory(); // 动态分配存储器

	void initial_condition();
	void initial_condition(int i); // 断电继续算

	void Paramentinput();	   // 孔喉数据导入函数声明
	void Paramentinput(int i); // 微孔非均匀文件读取
	void Para_cal_REV();	   //
	void Para_cal_REV_newton();
	void para_cal();				 // 喉道长度等相关参数计算
	void para_cal_in_newton();		 // 在牛顿迭代中计算 克努森数
	void para_cal(double);			 // 喉道长度等相关参数计算
	void para_cal_in_newton(double); // 在牛顿迭代中计算 克努森数

	double compre(double pressure); // 压缩系数
	double visco(double pressure, double z, double T);
	double micro_permeability(double pre);
	void Function_DS(double pressure);
	double Function_Slip(double knusen);
	double Function_Slip_clay(double knusen);

	void Eigen_solver(); // 瞬态扩散迭代求解流程
	void Eigen_solver_per();
	void Eigen_solver_per(double);
	void AMGXsolver(); // 产气
	void AMGX_permeability_solver();
	void AMGX_permeability_solver(double);
	void AMGX_solver_REV();
	void AMGX_solver_apparent_permeability_REV();

	void Matrix_gas_pro_REV();
	void Matrix_per_REV();
	void Matrix_permeability();
	void Matrix_permeability(double);
	void Matrix_gas_pro(); // 压力矩阵
	void CSR2COO();
	double Nor_inf(double A[]); // 误差

	void Eigen_subroutine_per(Eigen::SparseMatrix<double, Eigen::RowMajor> &, Eigen::VectorXd &);
	void Eigen_subroutine(Eigen::SparseMatrix<double, Eigen::RowMajor> &, Eigen::VectorXd &); // 非线性求解器
	void AMGX_solver_subroutine(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx);
	void AMGX_solver_subroutine_per(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx);

	double macro_outlet_flow();		 // 出口大孔流量
	double micro_outlet_free_flow(); // 出口微孔流量
	double micro_outlet_ad_flow();	 // 出口吸附量
	double macro_outlet_Q();		 // 出口大孔流量
	double micro_outlet_advec_Q();	 // 出口微孔流量
	double micro_outlet_diff_Q();	 // 出口吸附量
	double REV_OUT();

	double macro_mass_loss();
	double micro_free_mass_loss();
	double micro_ad_mass_loss();

	double clay_loss_per_step();
	double fracture_loss_per_step();
	double OM_HP_ad_loss_per_step();
	double OM_HP_free_loss_per_step();
	double OM_LP_ad_loss_per_step();
	double OM_LP_free_loss_per_step();

	void output(int n);			 // 输出VTK文件
	void output(int n, bool m);	 // REV 输出VTK 瞬态
	void output(int n, int m);	 // 渗透率计算输出vtk
	void output(double);		 // 十大攻关输出展示文件
	void output(double, double); // 单重孔网
	void mean_pore_size();

	~PNMsolver() // 析构函数，释放动态存储
	{
		delete[] dX, B;
		delete[] ia, ja, a, irn, jcn;
		delete[] Pb, Tb_in, Tb;
	}
};

void PNMsolver::mean_pore_size()
{
	memory();
	Paramentinput();
	initial_condition();
	para_cal();

	std::string filename("main_path_macro.txt");
	std::ifstream files(filename);
	std::set<int> pore_id_set;
	std::vector<double> pore_radius;

	for (size_t i = 0; i < 110; i++)
	{
		double waste(0);
		int id1(0);
		int id2(0);
		files >> id1 >> waste >> id2 >> waste;
		pore_id_set.insert(id1);
		pore_id_set.insert(id2);
	}

	files.close();

	ofstream outfiles("connected_radius");
	for (auto i : pore_id_set)
	{
		outfiles << Pb[i].Radiu / 1e-9 << endl;
	}

	int icounter{0};
	for (auto i : pore_id_set)
	{
		pore_radius.push_back(double(4) / double(3) * 3.14 * pow(Pb[i].Radiu, 3));
		icounter += 1;
	}
	double aver_volume{0};
	for (auto i : pore_radius)
	{
		aver_volume += i;
	}
	aver_volume = aver_volume / double(pore_id_set.size());

	double mean_radius = pow(double(3) / double(4) * aver_volume / 3.14, double(1) / double(3));
	cout << mean_radius / 1e-9 * 2 << " nm" << endl;
};

double PNMsolver::clay_loss_per_step()
{
	double clay_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		if (Pb[i].type == 0 || Pb[i].type == 1)
		{
			clay_loss_per_step += (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (Pb[i].compre_old * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
	}
	return (clay_loss_per_step);
}
double PNMsolver::fracture_loss_per_step()
{
	double fracture_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		if (Pb[i].type == 3)
		{
			fracture_loss_per_step += (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
	}
	return (fracture_loss_per_step);
}
double PNMsolver::OM_HP_ad_loss_per_step()
{
	double OM_HP_ad_loss_per_step = 0;

	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old + refer_pressure);
		double compre_2 = compre(Pb[i].pressure + refer_pressure);
		double n_ad1 = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
		if (Pb[i].type == 4)
		{
			OM_HP_ad_loss_per_step += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
	}
	return (OM_HP_ad_loss_per_step);
}
double PNMsolver::OM_HP_free_loss_per_step()
{
	double OM_HP_free_loss_per_step = 0;

	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old + refer_pressure);
		double compre_2 = compre(Pb[i].pressure + refer_pressure);
		double n_ad1 = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
		if (Pb[i].type == 4)
		{
			OM_HP_free_loss_per_step += (Pb[i].pressure_old + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
		}
	}
	return (OM_HP_free_loss_per_step);
}
double PNMsolver::OM_LP_ad_loss_per_step()
{
	double OM_LP_ad_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old + refer_pressure);
		double compre_2 = compre(Pb[i].pressure + refer_pressure);
		double n_ad1 = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
		if (Pb[i].type == 5)
		{
			OM_LP_ad_loss_per_step += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
	}
	return (OM_LP_ad_loss_per_step);
}
double PNMsolver::OM_LP_free_loss_per_step()
{
	double OM_LP_free_loss_per_step = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(Pb[i].pressure_old + refer_pressure);
		double compre_2 = compre(Pb[i].pressure + refer_pressure);
		double n_ad1 = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
		if (Pb[i].type == 5)
		{
			OM_LP_free_loss_per_step += (Pb[i].pressure_old + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
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
				rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
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
				rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
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
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
		Pb[i].visco_old = Pb[i].visco;
	}

	// Total gas content
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(inlet_pre + refer_pressure);
		double compre_2 = compre(outlet_pre + refer_pressure);

		double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));
		if (Pb[i].type == 0)
		{
			total_clay_HP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 1)
		{
			total_clay_LP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 2)
		{
			total_macropores += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 3)
		{
			total_fracture += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 4)
		{
			total_OM_HP_free += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_HP_ad += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
		else if (Pb[i].type == 5)
		{
			total_OM_LP_free += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_LP_ad += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
	}
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double compre_1 = compre(inlet_pre + refer_pressure);
		double compre_2 = compre(outlet_pre + refer_pressure);

		double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));
		if (Pb[i].type == 0)
		{
			total_clay_HP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 1)
		{
			total_clay_LP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 2)
		{
			total_macropores += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 3)
		{
			total_fracture += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		else if (Pb[i].type == 4)
		{
			total_OM_HP_free += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_HP_ad += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
		else if (Pb[i].type == 5)
		{
			total_OM_LP_free += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
			total_OM_LP_ad += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
		}
	}

	total_p = total_clay_HP + total_clay_LP + total_fracture + total_OM_HP_free + total_OM_HP_ad + total_OM_LP_free + total_OM_LP_ad;
	cout << "total_clay_HP = " << total_clay_HP << endl;
	cout << "total_clay_LP = " << total_clay_LP << endl;
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

		Rho_ID1 = 0.016 * (Pb[Tb_in[i].ID_1].pressure + refer_pressure) / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
		Rho_ID2 = 0.016 * (Pb[Tb_in[i].ID_2].pressure + refer_pressure) / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
		if (Pb[Tb_in[i].ID_1].type == 0) // clay_HP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_HP);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k;
		}
		else if (Pb[Tb_in[i].ID_1].type == 1) // clay_LP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_LP);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k;
		}
		else if (Pb[Tb_in[i].ID_1].type == 2) // crack
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 8;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // macro pores
		{
			Knusen_number_ID1 = 0;
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_1].type == 4) // OM_type_HP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_HP);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_1].pressure + refer_pressure), 2) / Rho_ID1;
		}
		else if (Pb[Tb_in[i].ID_1].type == 5) // OM_type_LP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_LP);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_1].pressure + refer_pressure), 2) / Rho_ID1;
		}

		// 6.96e-9
		if (Pb[Tb_in[i].ID_2].type == 0) // clay
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_HP);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k;
		}
		else if (Pb[Tb_in[i].ID_1].type == 1) // clay_LP
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_LP);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k;
		}
		else if (Pb[Tb_in[i].ID_2].type == 2) // crack
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 8;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // macro pores
		{
			Knusen_number_ID2 = 0;
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_2].type == 4) // OM_type1
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_HP);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_2].pressure + refer_pressure), 2) / Rho_ID2;
		}
		else if (Pb[Tb_in[i].ID_2].type == 5) // OM_type2
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_LP);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_2].pressure + refer_pressure), 2) / Rho_ID2;
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

	// ofstream outfile("conduct.txt");

	// for (size_t i = 0; i < 2 * tn; i++)
	// {
	// 	outfile << Tb_in[i].Conductivity << " ; " << Tb[i].Conductivity << endl;
	// }
	// outfile.close();
}

void PNMsolver::Para_cal_REV_newton()
{
	// 计算压缩系数 气体粘度
	for (int i = 0; i < pn; i++)
	{
		// Pb[i].compre = 0.702 * pow(M_E, -2.5 * Tr) * pow(Pr, 2) - 5.524 * pow(M_E, -2.5 * Tr) * Pr + 0.044 * pow(Tr, 2) - 0.164 * Tr + 1.15;
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
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

		Rho_ID1 = 0.016 * (Pb[Tb_in[i].ID_1].pressure + refer_pressure) / (Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature);
		Rho_ID2 = 0.016 * (Pb[Tb_in[i].ID_2].pressure + refer_pressure) / (Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature);
		if (Pb[Tb_in[i].ID_1].type == 0) // clay HP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_HP);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k;
		}
		else if (Pb[Tb_in[i].ID_1].type == 1) // clay LP
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_LP);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k;
		}
		else if (Pb[Tb_in[i].ID_1].type == 2) // macro pores
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_1].Radiu * 2);
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 8;
		}
		else if (Pb[Tb_in[i].ID_1].type == 3) // crack
		{
			Knusen_number_ID1 = 0;
			Slip_ID1 = Function_Slip_clay(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * pow(Pb[Tb_in[i].ID_1].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_1].type == 4) // OM_type1
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_HP);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_1].pressure + refer_pressure), 2) / Rho_ID1;
		}
		else if (Pb[Tb_in[i].ID_1].type == 5) // OM_type2
		{
			Knusen_number_ID1 = Pb[Tb_in[i].ID_1].visco / (Pb[Tb_in[i].ID_1].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_1].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_LP);
			Slip_ID1 = Function_Slip(Knusen_number_ID1);
			Apparent_K_ID1 = Slip_ID1 * Pb[Tb_in[i].ID_1].REV_k + Pb[Tb_in[i].ID_1].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_1].pressure + refer_pressure), 2) / Rho_ID1;
		}

		// 6.96e-9
		if (Pb[Tb_in[i].ID_2].type == 0) // clay_hp
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_HP);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k;
		}
		else if (Pb[Tb_in[i].ID_2].type == 1) // clay_lp
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_CLAY_LP);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k;
		}
		else if (Pb[Tb_in[i].ID_2].type == 2) // crack
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (Pb[Tb_in[i].ID_2].Radiu * 2);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 8;
		}
		else if (Pb[Tb_in[i].ID_2].type == 3)
		{
			Knusen_number_ID2 = 0;
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * pow(Pb[Tb_in[i].ID_2].Radiu, 2) / 12;
		}
		else if (Pb[Tb_in[i].ID_2].type == 4) // OM_type1
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_HP);
			Slip_ID2 = Function_Slip(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_2].pressure + refer_pressure), 2) / Rho_ID2;
		}
		else if (Pb[Tb_in[i].ID_2].type == 5) // OM_type2
		{
			Knusen_number_ID2 = Pb[Tb_in[i].ID_2].visco / (Pb[Tb_in[i].ID_2].pressure + refer_pressure) * sqrt(pi * Pb[Tb_in[i].ID_2].compre * 8.314 * Temperature / (2 * 0.016)) / (L_OM_LP);
			Slip_ID2 = Function_Slip_clay(Knusen_number_ID2);
			Apparent_K_ID2 = Slip_ID2 * Pb[Tb_in[i].ID_2].REV_k + Pb[Tb_in[i].ID_2].visco * Ds * n_max_ad * K_langmuir / pow(1 + K_langmuir * (Pb[Tb_in[i].ID_2].pressure + refer_pressure), 2) / Rho_ID2;
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

void PNMsolver::AMGX_solver_REV()
{
	auto start1 = high_resolution_clock::now();
	double acu_clay_HP{0}, acu_clay_LP{0}, acu_fracture{0}, acu_ad_OM_HP{0}, acu_free_OM_HP{0}, acu_ad_OM_LP{0}, acu_free_OM_LP{0}, acu_macro_pores{0};

	int n{1};
	int inter_n{0};								 // The interation of outer loop of Newton-raphoon method
	double total_loss = 0;						 // accumulation production
	ofstream outfile("Transient.txt", ios::app); // output permeability;

	Flag_eigen = false;
	Flag_Hybrid = false;
	memory();
	if (refer_pressure < 0.0001)
	{
		refer_pressure = 1e6;
	}

	Function_DS(inlet_pre + refer_pressure);
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

		if (Flag_outputvtk)
		{
			output(int(-2), bool(1));
		}
	}

	Para_cal_REV();
	Matrix_gas_pro_REV();
	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "solver.json"); // 200

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
			<< "acu_loss_clay_HP / total_p = " << acu_clay_HP / total_p << "\t"
			<< "acu_loss_clay_LP / total_p = " << acu_clay_LP / total_p << "\t"
			<< "acu_loss_fracture / total_p = " << acu_fracture / total_p << "\t"
			<< "acu_loss_macro / total_p = " << acu_macro_pores / total_p << "\t"
			<< "acu_loss_ad_OM_HP / total_p = " << acu_ad_OM_HP / total_p << "\t"
			<< "acu_loss_free_OM_HP / total_p = " << acu_free_OM_HP / total_p << "\t"
			<< "acu_loss_ad_OM_LP / total_p = " << acu_ad_OM_LP / total_p << "\t"
			<< "acu_loss_free_OM_LP / total_p = " << acu_free_OM_LP / total_p << "\t"

			<< "acu_loss_clay_HP = " << acu_clay_HP << "\t"
			<< "acu_loss_clay_LP = " << acu_clay_LP << "\t"
			<< "acu_loss_fracture = " << acu_fracture << "\t"
			<< "acu_loss_macro = " << acu_macro_pores << "\t"
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
	if (Flag_outputvtk)
	{
		output(time_step - 1, true); // 初始状态
	}

	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	AMGX_solver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	do
	{
		inter_n = 0;
		do
		{
			Para_cal_REV_newton();
			Matrix_gas_pro_REV();
			AMGX_solver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps && inter_n < force_out);

		time_all += dt;
		acu_clay_HP = 0;
		acu_clay_LP = 0;
		acu_fracture = 0;
		acu_free_OM_HP = 0;
		acu_free_OM_LP = 0;
		acu_ad_OM_HP = 0;
		acu_ad_OM_LP = 0;
		acu_macro_pores = 0;

		// acumu gas loss
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			double compre_2 = compre(Pb[i].pressure + refer_pressure);
			double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));

			if (Pb[i].type == 0)
			{
				acu_clay_HP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 1)
			{
				acu_clay_LP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 2)
			{
				acu_macro_pores += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 3)
			{
				acu_fracture += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 4)
			{
				acu_free_OM_HP += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_HP += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
			}
			else if (Pb[i].type == 5)
			{
				acu_free_OM_LP += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_LP += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
			}
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			double compre_2 = compre(Pb[i].pressure + refer_pressure);
			double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			double n_ad2 = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));

			if (Pb[i].type == 0)
			{
				acu_clay_HP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 1)
			{
				acu_clay_LP += (inlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 2)
			{
				acu_macro_pores += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 3)
			{
				acu_fracture += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
			}
			else if (Pb[i].type == 4)
			{
				acu_free_OM_HP += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_HP += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
			}
			else if (Pb[i].type == 5)
			{
				acu_free_OM_LP += (inlet_pre + refer_pressure) * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad1 / Rho_ad) + Pb[i].REV_porosity1) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * ((1 - Pb[i].REV_porosity2) * (Pb[i].REV_micro_porosity - n_ad2 / Rho_ad) + Pb[i].REV_porosity1) * 16 / (compre_2 * 8.314 * Temperature);
				acu_ad_OM_LP += Pb[i].volume * (1 - Pb[i].REV_porosity2) * (n_ad1 - n_ad2) * 1000;
			}
		}

		total_loss = acu_clay_HP + acu_clay_LP + acu_fracture + acu_free_OM_HP + acu_free_OM_LP + acu_ad_OM_HP + acu_ad_OM_LP;

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
				<< "machine_time = " << duration3.count() / 1000 + machine_time << "\t"
				<< "physical_time = " << time_all << "\t"
				<< "dt = " << dt << "\t"

				<< "total_loss / total_p = " << total_loss / total_p << "\t"
				<< "acu_loss_clay_HP / total_p = " << acu_clay_HP / total_p << "\t"
				<< "acu_loss_clay_LP / total_p = " << acu_clay_LP / total_p << "\t"
				<< "acu_loss_fracture / total_p = " << acu_fracture / total_p << "\t"
				<< "acu_loss_macro / total_p = " << acu_macro_pores / total_p << "\t"
				<< "acu_loss_ad_OM_HP / total_p = " << acu_ad_OM_HP / total_p << "\t"
				<< "acu_loss_free_OM_HP / total_p = " << acu_free_OM_HP / total_p << "\t"
				<< "acu_loss_ad_OM_LP / total_p = " << acu_ad_OM_LP / total_p << "\t"
				<< "acu_loss_free_OM_LP / total_p = " << acu_free_OM_LP / total_p << "\t"

				<< "acu_loss_clay_HP = " << acu_clay_HP << "\t"
				<< "acu_loss_clay_LP = " << acu_clay_LP << "\t"
				<< "acu_loss_fracture = " << acu_fracture << "\t"
				<< "acu_loss_macro = " << acu_macro_pores << "\t"
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

		if (inter_n <= double_time && dt < maximum_dt)
		{
			dt = dt * 2;
		}

		if (total_loss / total_p > 0.01 * n)
		{
			if (Flag_outputvtk)
			{
				output(time_step, true);
			}
			n++;
		}

		time_step++;
	} while (total_loss / total_p < 0.99);
	if (Flag_outputvtk)
	{
		output(time_step, true);
	}

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

void PNMsolver::Matrix_gas_pro_REV()
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
		if (Pb[i].type == 0 || Pb[i].type == 1) // 0 - clay HP; 4 - clay LP
		{
			B[i - inlet] = -0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 2 || Pb[i].type == 3) // 1 - crack; 5 - macro pores
		{
			B[i - inlet] = (-0.016) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 4 || Pb[i].type == 5) // 2 - OM-HP; 3 OM-LP
		{
			B[i - para_macro] = -0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - Pb[i].REV_porosity2) * Pb[i].volume / dt * ((1 - 0.016 * (Pb[i].pressure + refer_pressure) / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure))) - (1 - 0.016 * (Pb[i].pressure_old + refer_pressure) / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure))));
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
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
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

		if (Pb[i].type == 0 || Pb[i].type == 1) // clay HP
		{
			a[num + temp - 1] = 0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 2 || Pb[i].type == 3) // crack
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 4 || Pb[i].type == 5) // OM-HP
		{
			a[num + temp - 1] = 0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - Pb[i].REV_porosity2) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * (Pb[i].pressure + refer_pressure) - n_max_ad * 0.016 * pow(K_langmuir * (Pb[i].pressure + refer_pressure), 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * (Pb[i].pressure + refer_pressure)), 2));
		}

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
		if (Pb[i].type == 0 || Pb[i].type == 1) // 0 - clay HP; 4 - clay LP
		{
			B[i - inlet] = -0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 2 || Pb[i].type == 3) // 1 - crack; 5 - macro pores
		{
			B[i - inlet] = (-0.016) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt);
		}
		else if (Pb[i].type == 4 || Pb[i].type == 5) // 2 - OM-HP; 3 OM-LP
		{
			B[i - para_macro] = -0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt) - (1 - Pb[i].REV_porosity2) * Pb[i].volume / dt * ((1 - 0.016 * (Pb[i].pressure + refer_pressure) / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure))) - (1 - 0.016 * (Pb[i].pressure_old + refer_pressure) / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure))));
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
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
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
		if (Pb[i].type == 0 || Pb[i].type == 1)	 // clay HP
		{
			a[num + temp - 1] = 0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 2 || Pb[i].type == 3) // crack
		{
			a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值
		}
		else if (Pb[i].type == 4 || Pb[i].type == 5) // OM-HP
		{
			a[num + temp - 1] = 0.016 * (Pb[i].REV_porosity1 + (1 - Pb[i].REV_porosity2) * Pb[i].REV_micro_porosity) * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + (1 - Pb[i].REV_porosity2) * Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * (Pb[i].pressure + refer_pressure) - n_max_ad * 0.016 * pow(K_langmuir * (Pb[i].pressure + refer_pressure), 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * (Pb[i].pressure + refer_pressure)), 2));
		}

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 主对角线的初始值
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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

void PNMsolver::Eigen_solver_per(double i)
{
	refer_pressure = 0;
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													   // label of output file
	int inter_n{0};												   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;										   // accumulation production
	ofstream outfile("Intrinsic_Permeability_Eigen_BiCGSTAB.txt"); // output permeability;

	Flag_eigen = true;
	Flag_intri_per = true;
	memory();
	Paramentinput();
	initial_condition();
	para_cal(1);
	Matrix_permeability(1);

	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine_per(A0, B0);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start1);

	outfile << "permeability = " << (macro + micro_advec) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << " nD" << "\t"
			<< "pressure = " << inlet_pre + refer_pressure << " Pa" << "\t"
			<< "solve time = " << duration2.count() / 1000 << "s" << endl;
	if (Flag_outputvtk)
	{
		output(1, 1);
	}

	outfile.close();
}

// 升序多级排序，此时的判断条件是 < 号
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
			string eq_head = "=";
			string dot_head = ",";
			string mao_head = ";";
			string::size_type eq_idx{0};
			string::size_type dot_idx{0};
			string::size_type mao_idx{0};
			vector<int> iputings;
			getline(files, sline);
			assert(mao_idx = sline.find(mao_head) != string::npos);
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(dot_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				int ii;
				ss >> ii;
				iputings.push_back(ii);
				eq_idx++;
				dot_idx++;
			}
			istringstream ss(sline.substr(eq_idx + 1, mao_idx - eq_idx - 1));
			int ii;
			ss >> ii;
			iputings.push_back(ii);

			getline(files, sline);
			getline(files, sline);
			eq_idx = 0;
			dot_idx = 0;
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find("\t", dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				int ii;
				ss >> ii;
				iputings.push_back(ii);
				eq_idx++;
				dot_idx++;
			}

			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1));
				int ii;
				ss >> ii;
				iputings.push_back(ii);
				eq_idx++;
				dot_idx++;
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

			getline(files, sline);
			getline(files, sline);
			getline(files, sline);
			getline(files, sline);
			eq_idx = 0;
			dot_idx = 0;
			int iCounter{0};
			vector<double> oo;
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(mao_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				double ii;
				ss >> ii;
				oo.push_back(ii);
				eq_idx = 0;
				dot_idx = 0;
				getline(files, sline);
			}

			eq_idx = 0;
			dot_idx = 0;
			getline(files, sline);
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(mao_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				double ii;
				ss >> ii;
				oo.push_back(ii);
				eq_idx = 0;
				dot_idx = 0;
				getline(files, sline);
			}

			eq_idx = 0;
			dot_idx = 0;
			getline(files, sline);
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(mao_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				double ii;
				ss >> ii;
				oo.push_back(ii);
				eq_idx = 0;
				dot_idx = 0;
				getline(files, sline);
			}

			eq_idx = 0;
			dot_idx = 0;
			getline(files, sline);
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(mao_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				double ii;
				ss >> ii;
				oo.push_back(ii);
				eq_idx = 0;
				dot_idx = 0;
				getline(files, sline);
			}

			inlet_pre = oo[0];
			outlet_pre = oo[1];
			refer_pressure = oo[2];
			voxel_size = oo[3];
			domain_size_cubic = oo[4];
			dt = oo[5];

			porosity = oo[6];
			ko = oo[7];
			micro_radius = oo[8];
			// ko = porosity * pow(micro_radius, 2) / 32;
			cout << "ko = " << ko << endl;
			porosity_OMHP1 = oo[9];
			porosity_OMLP1 = oo[10];
			porosity_clay_HP1 = oo[11];
			porosity_clay_HP2 = oo[12];
			porosity_OMHP2 = porosity_OMHP1;
			porosity_OMLP2 = porosity_OMLP1;
			porosity_clay_HP2 = porosity_clay_HP1;
			porosity_clay_LP2 = porosity_clay_LP1;
			micro_porosity_HP = oo[13];
			micro_porosity_LP = oo[14];
			micro_porosity_Clay_HP = oo[15];
			micro_porosity_Clay_LP = oo[16];
			L_CLAY_HP = oo[17];
			L_CLAY_LP = oo[18];
			L_OM_HP = oo[19];
			L_OM_LP = oo[20];
			swww_clay = oo[21];
			swww_om = oo[22];
			K_OM_LP = oo[23];
			K_OM_HP = oo[24];
			K_Clay_HP = oo[25];
			K_Clay_LP = oo[26];

			eps = oo[27];
			maximum_dt = oo[28];
			force_out = int(oo[29]);
			double_time = oo[30];
			Flag_homo = oo[31];
			Flag_outputvtk = oo[32];
			domain_length = oo[33];

			a_OMLP = K_OM_LP / pow(Sw_max_OMLP, 2);
			K_OM_LP = a_OMLP * pow(Sw_OMLP - Sw_max_OMLP, 2);

			a_om = K_OM_HP / pow(Sw_max_om, 2);
			K_OM_HP = a_om * pow(Sw_om - Sw_max_om, 2);

			a_clay = K_Clay_HP / pow(Sw_max_clay, 6);
			K_Clay_HP = a_clay * pow(Sw_clay - Sw_max_clay, 6);

			if (Sw_clay >= 0.95)
			{
				K_Clay_HP = 1e-30;
			}

			if (Sw_om >= 0.95)
			{
				K_OM_HP = 1e-30;
				K_OM_LP = 1e-30;
			}
		}
	}

	macro_n = inlet + op + outlet;
	micro_n = m_inlet + mp + m_outlet;
	para_macro = inlet + outlet + m_inlet;
	NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

	if (flag == false)
	{
		cout << "voxel file missed!" << endl;
		abort();
	}

	cout << "pn = " << pn << endl;
	cout << "tn = " << tn << endl;
	cout << "inlet = " << inlet << "; " << "outlet = " << outlet << "; " << "m_inlet = " << m_inlet << "; " << "m_outlet = " << m_outlet << "; " << "op = " << op << "; " << "mp = " << mp << "; " << endl;

	dX = new double[op + mp];
	B = new double[op + mp];

	ia = new int[op + mp + 1];
	ja = new int[NA];

	irn = new int[NA];
	jcn = new int[NA];
	a = new double[NA];

	Pb = new pore[pn];
	Tb_in = new throat[2 * tn];
	Tb = new throatmerge[2 * tn];
}

void PNMsolver::initial_condition()
{
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		Pb[i].pressure = inlet_pre; //- double(double(i) / double(pn) * 100)
		Pb[i].pressure_old = Pb[i].pressure;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		Pb[i].pressure = outlet_pre;
		Pb[i].pressure_old = outlet_pre;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = pn - m_outlet; i < pn; i++)
	{
		Pb[i].pressure = outlet_pre;
		Pb[i].pressure_old = outlet_pre;
	}
#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("initial_condition diff = %.16g\n",
		   end - start);
#endif
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

			if (Flag_Hybrid == true)
			{
				if (Flag_homo == true)
				{
					cout << "亚分辨区域均质" << "Km = " << ko << endl;
					for (int i = 0; i < pn; i++)
					{
						double waste{0};
						porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >> Pb[i].porosity >> Pb[i].km;
						Pb[i].km = ko;
						Pb[i].porosity = porosity;
						Pb[i].radius_micro = micro_radius;
					}
				}
				else
				{
					cout << "亚分辨区域非均质" << endl;
					for (int i = 0; i < pn; i++)
					{
						double waste{0};
						porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].radius_micro >> Pb[i].porosity >> Pb[i].km;
					}
				}
			}
			else
			{
				if (Flag_homo == true)
				{
					cout << "REV 假设每类矿物均质" << endl;
					for (int i = 0; i < pn; i++)
					{
						double waste{0};
						// porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].REV_porosity1 >> Pb[i].radius_micro >> Pb[i].REV_k; // REV
						porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum; // REV
						if (Pb[i].type == 0)																							// CLAY-HP																					// CLAY-HP																							// 粘土
						{
							Pb[i].REV_porosity1 = porosity_clay_HP1;
							Pb[i].REV_porosity2 = porosity_clay_HP2;
							Pb[i].REV_k = K_Clay_HP;
							Pb[i].REV_micro_porosity = micro_porosity_Clay_HP;
						}
						else if (Pb[i].type == 1) // CLAY-LP
						{
							Pb[i].REV_porosity1 = porosity_clay_LP1;
							Pb[i].REV_porosity2 = porosity_clay_LP2;
							Pb[i].REV_k = K_Clay_LP;
							Pb[i].REV_micro_porosity = micro_porosity_Clay_LP;
						}
						else if (Pb[i].type == 2) // 大孔
						{
						}
						else if (Pb[i].type == 3) // 裂缝
						{
						}
						else if (Pb[i].type == 4) // 联通有机质
						{
							Pb[i].REV_porosity1 = porosity_OMHP1;
							Pb[i].REV_porosity2 = porosity_OMHP2;
							Pb[i].REV_k = K_OM_HP;
							Pb[i].REV_micro_porosity = micro_porosity_HP;
						}
						else if (Pb[i].type == 5) // 不联通有机质
						{
							Pb[i].REV_porosity1 = porosity_OMLP1;
							Pb[i].REV_porosity2 = porosity_OMLP2;
							Pb[i].REV_k = K_OM_LP;
							Pb[i].REV_micro_porosity = micro_porosity_LP;
						}
					}
				}
				else
				{
					cout << "REV 假设每类矿物非均质" << endl;
					for (int i = 0; i < pn; i++)
					{
						double waste{0};
						porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> Pb[i].REV_porosity1 >> Pb[i].radius_micro >> Pb[i].REV_k; // REV
						Pb[i].REV_porosity2 = Pb[i].REV_porosity1;
						Pb[i].REV_micro_porosity = 0.1;
					}
				}
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

			if (Flag_Hybrid == true)
			{
				for (int i = 0; i < 2 * tn; i++)
				{
					throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
				}
			}
			else
			{
				for (int i = 0; i < 2 * tn; i++)
				{
					throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].Length;
				}
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
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
		Pb[i].visco_old = Pb[i].visco;
	}


	for (int i = inlet; i < macro_n - outlet; i++)
	{
		double compre_1 = compre(inlet_pre + refer_pressure);
		double compre_2 = compre(outlet_pre + refer_pressure);
		total_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
	}

	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double compre_1 = compre(inlet_pre + refer_pressure);
		double compre_2 = compre(outlet_pre + refer_pressure);

		double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
		double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));

		total_micro_free += (inlet_pre + refer_pressure) * (Pb[i].porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
		total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
	}
	total_p = total_macro + total_micro_free + total_micro_ad;
	cout << "total_macro = " << total_macro << endl;
	cout << "total_micro_free = " << total_micro_free << endl;
	cout << "total_micro_ad = " << total_micro_ad << endl;
	cout << "total_p = " << total_p << endl;

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
		// 计算克努森数
		double Knusen_number{0};
		double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
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

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;

	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);
#endif
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
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].compre_old = Pb[i].compre;
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
		Pb[i].visco_old = Pb[i].visco;
	}

	// Total gas content
	double compre_1 = compre(inlet_pre + refer_pressure);
	double compre_2 = compre(outlet_pre + refer_pressure);
	double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
	double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));

	for (int i = inlet; i < macro_n - outlet; i++)
	{
		total_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
	}

	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		total_micro_free += (inlet_pre + refer_pressure) * (Pb[i].porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
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
		Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
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
#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);
#endif
}

void PNMsolver::para_cal_in_newton(double mode)
{
	// 计算压缩系数
	for (int i = 0; i < pn; i++)
	{
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量
	for (int i = 0; i < 2 * tn; i++)
	{
		// 计算克努森数
		double Knusen_number{0};
		double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
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
	// // full_coord
	// #pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
	// 	for (int i = 0; i < pn; i++)
	// 	{
	// 		Pb[i].full_coord = 0;
	// 		Pb[i].full_accum = 0;
	// 	}
	// #pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
	// 	for (int i = 0; i <= label; i++)
	// 	{
	// 		Pb[Tb[i].ID_1].full_coord += 1;
	// 	}

	// // full_accum
	// 	Pb[0].full_accum = Pb[0].full_coord;
	// #pragma omp parallel for num_threads(int(omp_get_max_threads() * 0.8))
	// 	for (int i = 1; i < pn; i++)
	// 	{
	// 		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	// 	}
}

void PNMsolver::Matrix_gas_pro()
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
		B[i - inlet] = -0.016 * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt);
		temp = 0, temp1 = 0;
		num = ia[i - inlet];
		// macropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet) // 进口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
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

		ja[num + temp - 1] = i - inlet + 1;													  // 第i行对角线的值的位置
		a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
		B[i - para_macro] = -0.016 * Pb[i].porosity * Pb[i].volume * ((Pb[i].pressure + refer_pressure) / Pb[i].compre - (Pb[i].pressure_old + refer_pressure) / Pb[i].compre_old) / (8.314 * Temperature * dt) - Pb[i].volume / dt * ((1 - 0.016 * (Pb[i].pressure + refer_pressure) / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure))) - (1 - 0.016 * (Pb[i].pressure_old + refer_pressure) / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure))));

		temp = 0, temp1 = 0;
		num = ia[i - para_macro];
		// micropore
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进出口边界
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));

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
		a[num + temp - 1] = 0.016 * Pb[i].porosity * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * (Pb[i].pressure + refer_pressure) - n_max_ad * 0.016 * pow(K_langmuir * (Pb[i].pressure + refer_pressure), 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * (Pb[i].pressure + refer_pressure)), 2));
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) // 主对角线的初始值
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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

						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (-(Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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

		// ofstream ia_out("ia_out_amgx.txt");
		// for (size_t i = 0; i < op + mp + 1; i++)
		// {
		// 	ia_out << ia[i] << endl;
		// }

		// ofstream ja_out("ja_out_amgx.txt");
		// for (size_t i = 0; i < NA; i++)
		// {
		// 	ja_out << ja[i] << endl;
		// }

		// ofstream a_out("a_out_amgx.txt");
		// for (size_t i = 0; i < NA; i++)
		// {
		// 	a_out << a[i] << endl;
		// }
	}

	// ofstream ia_out("ia_out_EIGEN.txt");
	// for (size_t i = 0; i < op + mp + 1; i++)
	// {
	// 	ia_out << ia[i] << endl;
	// }

	// ofstream ja_out("ja_out_EIGEN.txt");
	// for (size_t i = 0; i < NA; i++)
	// {
	// 	ja_out << ja[i] << endl;
	// }

	// ofstream a_out("a_out_EIGEN.txt");
	// for (size_t i = 0; i < NA; i++)
	// {
	// 	a_out << a[i] << endl;
	// }
}

void PNMsolver::Eigen_subroutine(Eigen::SparseMatrix<double, Eigen::RowMajor> &A0, Eigen::VectorXd &B0)
{
	using namespace Eigen;
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
		// cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + inlet + outlet + m_inlet].pressure += x[i];
	}
	////outlet部分孔设置为解吸出口
	for (int i = 0; i < inlet; i++)
	{
		Pb[i].pressure += x[Tb[i].ID_2 - inlet];
	}
	for (int i = macro_n; i < macro_n + m_inlet; i++)
	{
		Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
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

	if (Flag_intri_per == true)
	{
		// 更新应力场
		for (int i = inlet; i < inlet + op; i++)
		{
			Pb[i].pressure = x[i - inlet];
		}
		for (int i = op; i < op + mp; i++)
		{
			Pb[i + inlet + outlet + m_inlet].pressure = x[i];
		}
	}
	else
	{
		// 更新应力场
		for (int i = inlet; i < inlet + op; i++)
		{
			Pb[i].pressure += x[i - inlet];
		}
		for (int i = op; i < op + mp; i++)
		{
			Pb[i + inlet + outlet + m_inlet].pressure += x[i];
		}
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Eigen subroutine Time-consuming = " << duration.count() / 1000 << "S" << endl;
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
				rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
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
				rho = (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature); // rho mol/m^3
			}
			else
			{
				rho = (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature); // rho mol/m^3
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
			Q_outlet += dt * Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure))) * 1000;
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
																								  // out_flux << Pb[Tb[j].ID_1].pressure << "  " << Pb[Tb[j].ID_2].pressure << "  " << abs(Q_outlet) << "  " << 1 << "  " << i << endl;
		}
	}
	return abs(Q_outlet);
}; // 出口大孔流量

double PNMsolver::micro_outlet_advec_Q()
{
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			Q_outlet += (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity; // 体积流量
		}
	}
	return abs(Q_outlet);
}; // 出口微孔流量

double PNMsolver::micro_outlet_diff_Q()
{
	double Q_outlet = 0;
	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			double average_density = ((Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) + (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature)) / 2;
			Q_outlet += Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure))) / (average_density * 16e-3); // 体积流量
		}
	}
	return abs(Q_outlet);
}; // 出口吸附量

double PNMsolver::macro_mass_loss()
{
	double macro_mass_loss = 0;
	for (int i = inlet; i < macro_n - outlet; i++)
	{
		macro_mass_loss += (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
	}
	return (macro_mass_loss);
}

double PNMsolver::micro_free_mass_loss()
{
	double micro_mass_loss = 0;
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));		  // kg/m3
		double n_ad_old = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure))); // kg/m3

		micro_mass_loss += (Pb[i].porosity - n_ad_old / Rho_ad) * (Pb[i].pressure_old + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre_old * 8.314 * Temperature) - (Pb[i].porosity - n_ad_new / Rho_ad) * (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
	}
	return (micro_mass_loss);
}

double PNMsolver::micro_ad_mass_loss() // - outlet_pre * Pb[i].volume * (porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature)
{
	double micro_ad_mass_loss = 0;
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
		double n_ad_old = n_max_ad * (K_langmuir * (Pb[i].pressure_old + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure_old + refer_pressure)));

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
	ofstream flux("flux.txt");
	vector<double> macro_fluxes;
	vector<double> micro_fluxes;
	vector<double> inter_fluxes;

	ostringstream name;
	name << "Permeability";
	name << to_string(inlet_pre + refer_pressure);
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

	outfile << "SCALARS free_gas_flux double 1" << endl;
	outfile << "LOOKUP_TABLE table12" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
		// flux << kkk << endl;
		outfile << kkk << endl;
		if (Pb[Tb[i].ID_1].type == 0 && Pb[Tb[i].ID_2].type == 0) // 高岭石流量
		{
			macro_fluxes.push_back(kkk);
		}
		else if (Pb[Tb[i].ID_1].type == 1 && Pb[Tb[i].ID_2].type == 1) // 绿泥石流量
		{
			micro_fluxes.push_back(kkk);
		}
		else
		{
			inter_fluxes.push_back(kkk);
		}
	}

	auto macro_ptr = max_element(macro_fluxes.begin(), macro_fluxes.end());
	auto micro_ptr = max_element(micro_fluxes.begin(), micro_fluxes.end());
	auto inter_ptr = max_element(inter_fluxes.begin(), inter_fluxes.end());
	auto macro_min_ptr = min_element(macro_fluxes.begin(), macro_fluxes.end());
	auto micro_min_ptr = min_element(micro_fluxes.begin(), micro_fluxes.end());
	auto inter_min_ptr = min_element(inter_fluxes.begin(), inter_fluxes.end());
	auto thred1 = *macro_ptr * 0.1;
	auto thred2 = *micro_ptr * 0.1;
	auto thred3 = *inter_min_ptr * 0.1;

	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
		/*输出流量分布*/
		if (Pb[Tb[i].ID_1].type == 0 && Pb[Tb[i].ID_2].type == 0 && kkk >= thred1) // 高岭石 主通道
		{
			Tb[i].main_free = int(0);
			// main_path_macro << Tb[i].ID_1 << "\t" << Pb[Tb[i].ID_1].Radiu << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_2].Radiu << endl;
		}
		else if (Pb[Tb[i].ID_1].type == 1 && Pb[Tb[i].ID_2].type == 1 && kkk >= thred2) // 绿泥石 主通道
		{
			Tb[i].main_free = int(1);
			// main_path_micro << i << ";" << kkk << endl;
		}
		else if ((Pb[Tb[i].ID_1].type == 1 && Pb[Tb[i].ID_2].type == 0 && kkk >= thred2) || (Pb[Tb[i].ID_1].type == 0 && Pb[Tb[i].ID_2].type == 1 && kkk >= thred3))
		{
			Tb[i].main_free = int(3);
		}
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
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n)
	// 	{
	// 		micro_fluxes.push_back(kkk);
	// 	}
	// }

	// auto macro_ptr = max_element(macro_fluxes.begin(), macro_fluxes.end());
	// auto micro_ptr = max_element(micro_fluxes.begin(), micro_fluxes.end());
	// auto macro_min_ptr = min_element(macro_fluxes.begin(), macro_fluxes.end());
	// auto micro_min_ptr = min_element(micro_fluxes.begin(), micro_fluxes.end());
	// auto thred1 = *macro_ptr * 0.1;
	// auto thred2 = *micro_ptr * 0.1;

	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
	// 	/*输出流量分布*/
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk >= thred1) // 大孔 主通道
	// 	{
	// 		Tb[i].main_free = int(0);
	// 		// main_path_macro << Tb[i].ID_1 << "\t" << Pb[Tb[i].ID_1].Radiu << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_2].Radiu << endl;
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
	// auto Thred3 = *macro_diff_ptr * 0.1;
	// auto Thred4 = *diff_ptr * 0.1;
	// flux << "macro_max: " << *macro_ptr << endl
	// 	 << "macro_min: " << *macro_min_ptr << endl
	// 	 << "micro_max: " << *micro_ptr << endl
	// 	 << "micro_min: " << *micro_min_ptr << endl;

	// flux << "diff_max: " << *diff_ptr << endl;
	// /*输出流量分布*/
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
	// 	else if (Tb[i].ID_1 >= macro_n && Tb[i].ID_2 >= macro_n)
	// 	{
	// 		micro_fluxes.push_back(kkk);
	// 	}
	// }

	// auto macro_ptr = max_element(macro_fluxes.begin(), macro_fluxes.end());
	// auto micro_ptr = max_element(micro_fluxes.begin(), micro_fluxes.end());
	// auto macro_min_ptr = min_element(macro_fluxes.begin(), macro_fluxes.end());
	// auto micro_min_ptr = min_element(micro_fluxes.begin(), micro_fluxes.end());
	// auto thred1 = *macro_ptr * 0.1;
	// auto thred2 = *micro_ptr * 0.1;

	// for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	double kkk = Tb[i].Conductivity * abs(Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure);
	// 	/*输出流量分布*/
	// 	if (Tb[i].ID_1 < macro_n && Tb[i].ID_2 < macro_n && kkk >= thred1) // 大孔 主通道
	// 	{
	// 		Tb[i].main_free = int(0);
	// 		// main_path_macro << Tb[i].ID_1 << "\t" << Pb[Tb[i].ID_1].Radiu << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_2].Radiu << endl;
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
	// auto Thred3 = *macro_diff_ptr * 0.1;
	// auto Thred4 = *diff_ptr * 0.1;
	// flux << "macro_max: " << *macro_ptr << endl
	// 	 << "macro_min: " << *macro_min_ptr << endl
	// 	 << "micro_max: " << *micro_ptr << endl
	// 	 << "micro_min: " << *micro_min_ptr << endl;

	// flux << "diff_max: " << *diff_ptr << endl;
	// /*输出流量分布*/
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

	// main_path_macro.close();
	// main_path_micro.close();
	// main_path_double.close();
	// sub_path_macro.close();
	// sub_path_micro.close();
	// sub_path_double.close();
	outfile.close();
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
			auto n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			auto n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));
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
			auto compre_1 = compre(inlet_pre + refer_pressure);
			auto compre_2 = compre(outlet_pre + refer_pressure);
			auto n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			auto n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));
			auto out = (inlet_pre + refer_pressure) * (Pb[i].porosity - n_ad1 / Rho_ad) * Pb[i].volume * 0.016 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].porosity - n_ad2 / Rho_ad) * 0.016 / (compre_2 * 8.314 * Temperature);
			outfile << out << endl;
		}
		else
		{
			auto compre_1 = compre(inlet_pre + refer_pressure);
			auto compre_2 = compre(outlet_pre + refer_pressure);
			auto out = (inlet_pre + refer_pressure) * Pb[i].volume * 0.016 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 0.016 / (compre_2 * 8.314 * Temperature);
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

	Function_DS(inlet_pre + refer_pressure);
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
	Matrix_gas_pro();

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
	if (Flag_outputvtk)
	{
		output(-1);
	}

	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine(A0, B0);
	do
	{
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_gas_pro();
			Eigen_subroutine(A0, B0);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step << endl;
			cout << endl;
		} while (norm_inf > eps && inter_n < force_out);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			acu_flow_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre + refer_pressure);
			double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
			double n_ad_old = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			acu_free_micro += (Pb[i].porosity - n_ad_old / Rho_ad) * (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (Pb[i].porosity - n_ad_new / Rho_ad) * (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)) - K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure))); // 微孔累计产气质量 单位g
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

		if (inter_n < double_time && dt < maximum_dt)
		{
			dt = dt * 2;
		}

		if (total_flow / total_p > 0.1 * n)
		{
			if (Flag_outputvtk)
			{
				output(time_step);
			}

			n++;
		}

		time_step++;
	} while (total_flow / total_p < 0.99);
	if (Flag_outputvtk)
	{
		output(time_step);
	}

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
	Flag_intri_per = false;
	Function_DS(inlet_pre + refer_pressure);
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

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< (inlet_pre + refer_pressure) / 1e6 << "\t"
				<< duration2.count() / 1000 << "s" << "\t"
				<< endl;
		refer_pressure = i * 1e6;
		Function_DS(inlet_pre + refer_pressure);
		initial_condition();
	}
	outfile.close();
}

void PNMsolver::AMGX_solver_subroutine(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx)
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

void PNMsolver::AMGX_solver_subroutine_per(AMGX_matrix_handle &A_amgx, AMGX_vector_handle &b_amgx, AMGX_vector_handle &solution_amgx, AMGX_solver_handle &solver, int n_amgx, int nnz_amgx)
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

	if (Flag_intri_per == true)
	{
		// 更新应力场
		for (int i = inlet; i < inlet + op; i++)
		{
			Pb[i].pressure = dX[i - inlet];
		}
		for (int i = op; i < op + mp; i++)
		{
			Pb[i + inlet + outlet + m_inlet].pressure = dX[i];
		}
	}
	else
	{
		// 更新应力场
		for (int i = inlet; i < inlet + op; i++)
		{
			Pb[i].pressure += dX[i - inlet];
		}
		for (int i = op; i < op + mp; i++)
		{
			Pb[i + inlet + outlet + m_inlet].pressure += dX[i];
		}
	}

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
	int inter_n{0};								 // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;						 // accumulation production
	ofstream outfile("Transient.txt", ios::app); // output permeability;

	Function_DS(inlet_pre + refer_pressure);
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
	Matrix_gas_pro();

	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "solver.json"); // 200

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

	if (Flag_outputvtk)
	{
		output(time_step - 1); // 初始状态
	}

	// end AMGX initialization
	// ************ begin AMGX solver ************
	int nn{1};
	AMGX_solver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	do
	{
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_gas_pro();
			AMGX_solver_subroutine(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			// MKLsolve();
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "dt = " << dt << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t"
				 << "outer loop = " << time_step + 1 << endl;
			cout << endl;
		} while (norm_inf > eps && inter_n < force_out);

		time_all += dt;
		acu_flow_macro = 0;
		acu_free_micro = 0;
		acu_ad_micro = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			acu_flow_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_old = compre(inlet_pre + refer_pressure);
			double n_ad_new = n_max_ad * (K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure)));
			double n_ad_old = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			acu_free_micro += (Pb[i].porosity - n_ad_old / Rho_ad) * (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_old * 8.314 * Temperature) - (Pb[i].porosity - n_ad_new / Rho_ad) * (Pb[i].pressure + refer_pressure) * Pb[i].volume * 16 / (Pb[i].compre * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			acu_ad_micro += Pb[i].volume * n_max_ad * 1000 * abs(K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)) - K_langmuir * (Pb[i].pressure + refer_pressure) / (1 + K_langmuir * (Pb[i].pressure + refer_pressure))); // 微孔累计产气质量 单位g
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

		if (inter_n < double_time && dt < maximum_dt)
		{
			dt = dt * 2;
		}

		if (total_flow / total_p > 0.1 * n)
		{
			if (Flag_outputvtk)
			{
				output(time_step);
			}

			n++;
		}

		time_step++;
	} while (total_flow / total_p < 0.99);

	if (Flag_outputvtk)
	{
		output(time_step);
	}

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
#ifdef _OPENMP
	double start = omp_get_wtime();
#endif
	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	int temp2 = 0;

	ia[0] = 1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
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
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
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
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));

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
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (-(Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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

						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (-(Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
		for (size_t i = 0; i < op + mp + 1; i++)
		{
			ia[i] += -1;
		}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
		for (size_t i = 0; i < ia[op + mp]; i++)
		{
			ja[i] += -1;
		}
	}

#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("matrix diff = %.16g\n",
		   end - start);
#endif
};

void PNMsolver::para_cal_in_newton()
{
// 计算压缩系数
#ifdef _OPENMP
	double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < pn; i++)
	{
		Pb[i].compre = compre(Pb[i].pressure + refer_pressure);
		Pb[i].visco = visco(Pb[i].pressure + refer_pressure, Pb[i].compre, Temperature);
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, temp11 = 0, temp22 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
		// 计算克努森数
		double Knusen_number{0};
		double Average_pressure = (Pb[Tb_in[i].ID_1].pressure + Pb[Tb_in[i].ID_2].pressure + refer_pressure + refer_pressure) / 2;
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

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;
	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("para_cal diff = %.16g\n",
		   end - start);
#endif
}

void PNMsolver::AMGX_permeability_solver()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													  // label of output file
	int inter_n{0};												  // The interation of outer loop of Newton-raphoon method
	double total_flow = 0;										  // accumulation production
	ofstream outfile("apparent_Permeability_amgx.txt", ios::app); // output permeability;

	Flag_eigen = false;
	Flag_intri_per = false;
	Function_DS(inlet_pre + refer_pressure);
	memory();

	// refer_pressure = 1e6;
	Paramentinput();
	initial_condition();
	para_cal();
	Matrix_permeability();

	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "solver.json"); // for 505050

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
	AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	for (size_t i = 2; i < 52; i++)
	{
		para_cal_in_newton();
		Matrix_permeability();
		AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
		inter_n = 0;
		do
		{
			para_cal_in_newton();
			Matrix_permeability();
			AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
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

		outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
				<< (inlet_pre + refer_pressure) / 1e6 << "\t"
				<< duration2.count() / 1000 << "s" << "\t"
				<< endl;
		refer_pressure += 1e6;
		Function_DS(inlet_pre + refer_pressure);
		initial_condition();
		total_macro = 0;
		total_micro_free = 0;
		total_micro_ad = 0;
		for (int i = inlet; i < macro_n - outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			double compre_2 = compre(outlet_pre + refer_pressure);
			total_macro += (inlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * 16 / (compre_2 * 8.314 * Temperature);
		}
		for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
		{
			double compre_1 = compre(inlet_pre + refer_pressure);
			double compre_2 = compre(outlet_pre + refer_pressure);

			double n_ad1 = n_max_ad * (K_langmuir * (inlet_pre + refer_pressure) / (1 + K_langmuir * (inlet_pre + refer_pressure)));
			double n_ad2 = n_max_ad * (K_langmuir * (outlet_pre + refer_pressure) / (1 + K_langmuir * (outlet_pre + refer_pressure)));

			total_micro_free += (inlet_pre + refer_pressure) * (Pb[i].porosity - n_ad1 / Rho_ad) * Pb[i].volume * 16 / (compre_1 * 8.314 * Temperature) - (outlet_pre + refer_pressure) * Pb[i].volume * (Pb[i].porosity - n_ad2 / Rho_ad) * 16 / (compre_2 * 8.314 * Temperature);
			total_micro_ad += Pb[i].volume * (n_ad1 - n_ad2) * 1000;
		}
	}

	// para_cal_in_newton();
	// Matrix_permeability();
	// AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	// inter_n = 0;
	// do
	// {
	// para_cal_in_newton();
	// Matrix_permeability();
	// AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	// inter_n++;
	// cout << "Inf_norm = " << norm_inf << "\t\t"
	//  << "inner loop = " << inter_n
	//  << "\t\t" << endl;
	// } while (norm_inf > eps_per);
	//
	// macro = macro_outlet_Q();
	// micro_advec = micro_outlet_advec_Q();
	// micro_diff = micro_outlet_diff_Q();
	//
	// auto stop2 = high_resolution_clock::now();
	// auto duration2 = duration_cast<milliseconds>(stop2 - start1);
	//
	// outfile << (macro + micro_advec + micro_diff) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << "\t"
	// << (inlet_pre + refer_pressure) / 1e6 << "\t"
	// << duration2.count() / 1000 << "s" << "\t"
	// << endl;
	// outfile.close();

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

void PNMsolver::Matrix_permeability(double mode)
{
#ifdef _OPENMP
	double start = omp_get_wtime();
#endif
	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	int temp2 = 0;

	ia[0] = 1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
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
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			// printf("i = %d,j = %d, I am Thread %d\n", i, j, omp_get_thread_num());
			if (Tb[j].ID_2 < inlet) // 进口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
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
		num1 += 1;					  // 加上对角线的非0值
		ia[i - inlet + 1] = num1 + 1; // 前i行累计的非零值数量，其中1为ia[0]的值

		ja[num + temp - 1] = i - inlet + 1; // 第i行对角线的值的位置
		a[num + temp - 1] = 0;				// 主对角线的初始值

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			// printf("i = %d,j = %d, I am Thread %d\n", i, j, omp_get_thread_num());
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
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			// printf("i = %d,j = %d, I am Thread %d\n", i, j, omp_get_thread_num());
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进出口边界
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure);
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
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
		for (size_t i = 0; i < op + mp + 1; i++)
		{
			ia[i] += -1;
		}

#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
		for (size_t i = 0; i < ia[op + mp]; i++)
		{
			ja[i] += -1;
		}
	}

#ifdef _OPENMP
	double end = omp_get_wtime();
	printf("matrix diff = %.16g\n",
		   end - start);
#endif
};

void PNMsolver::Matrix_per_REV()
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
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - inlet] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
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
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num - 1] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
						a[num] += Tb[j].Conductivity * (-0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre);
					}
					else
					{

						a[ia[i - inlet] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre);
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
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));
			}
			else
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (Pb[Tb[j].ID_1].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				else
				{
					B[Tb[j].ID_1 - para_macro] += -0.016 * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (Pb[Tb[j].ID_2].compre * 8.314 * Temperature) * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure);
				}
				B[Tb[j].ID_1 - para_macro] += -Tb[j].Surface_diff_conduc * (K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure)) - K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure)));

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
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre); // 对角线
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre); // 对角线
				}
				a[ia[i - para_macro] + temp - 1] += Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2);
			}
			else
			{
				if (temp1 < temp) // 下三角
				{
					if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure)
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * (-(Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num - 1] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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

						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (2 * (Pb[Tb[j].ID_1].pressure + refer_pressure) - (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * (-(Pb[Tb[j].ID_1].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_1].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
					}
					else
					{
						a[ia[i - para_macro] + temp - 1] += 0.016 * Tb[j].Conductivity * (Pb[Tb[j].ID_2].pressure + refer_pressure) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * K_langmuir / pow(1 + K_langmuir * (Pb[Tb[j].ID_1].pressure + refer_pressure), 2); // 对角线
						a[num] += 0.016 * Tb[j].Conductivity * ((Pb[Tb[j].ID_1].pressure + refer_pressure) - 2 * (Pb[Tb[j].ID_2].pressure + refer_pressure)) / (8.314 * Temperature * Pb[Tb[j].ID_2].compre) + Tb[j].Surface_diff_conduc * (-K_langmuir) / pow(1 + K_langmuir * (Pb[Tb[j].ID_2].pressure + refer_pressure), 2);
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

void PNMsolver::AMGX_solver_apparent_permeability_REV()
{
	auto start1 = high_resolution_clock::now();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;			   // label of output file
	int inter_n{0};		   // The interation of outer loop of Newton-raphoon method
	double total_flow = 0; // accumulation production
	string outputfilename(to_string(Sw_om));
	ofstream outfile("apparent_Permeability_amgx_OM1_Sw" + outputfilename + "_.txt", ios::app); // output permeability;

	Flag_eigen = false;
	Flag_intri_per = false;
	Flag_Hybrid = false;
	memory();

	Function_DS(inlet_pre + refer_pressure);
	Paramentinput();
	initial_condition();
	Para_cal_REV();
	Matrix_per_REV();

	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "solver.json"); // for 505050
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
	AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
	for (size_t i = 2; i < 52; i++)
	{
		Para_cal_REV_newton();
		Matrix_per_REV();
		AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
		inter_n = 0;
		do
		{
			Para_cal_REV_newton();
			Matrix_per_REV();
			AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);
			inter_n++;
			cout << "Inf_norm = " << norm_inf << "\t\t"
				 << "inner loop = " << inter_n
				 << "\t\t" << endl;
		} while (norm_inf > eps_per);
		// output(i, 1);

		macro = macro_outlet_Q();
		micro_advec = micro_outlet_advec_Q();

		auto stop2 = high_resolution_clock::now();
		auto duration2 = duration_cast<milliseconds>(stop2 - start1);

		outfile << (macro + micro_advec) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-15 << " mD\t"
				<< (inlet_pre + refer_pressure) / 1e6 << " MPa"
				<< endl;

		// outfile << (macro + micro_advec) * 1.763e-5 * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-15 << "\t"
		// 		<< (inlet_pre + refer_pressure) / 1e6
		// 		<< endl;
		if (Flag_outputvtk)
		{
			output(1, 1);
		}

		refer_pressure += 0.1e6;
		Function_DS(inlet_pre + refer_pressure);
		initial_condition();
	}
	outfile << "K_OM_HP = " << K_OM_HP << " K_OM_LP = " << K_OM_LP << " K_CLAY_HP = " << K_Clay_HP << " K_CLAY_LP = " << K_Clay_LP << "\t";
	outfile << "L_OM_HP = " << L_OM_HP << " L_OM_LP = " << L_OM_LP << " L_CLAY_HP = " << L_CLAY_HP << " L_CLAY_LP = " << L_CLAY_LP;

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
}

void PNMsolver::AMGX_permeability_solver(double mode)
{
	refer_pressure = 0;
	double start = omp_get_wtime();
	double macro{0}, micro_advec{0}, micro_diff{0};

	int n = 1;													   // label of output file
	double total_flow = 0;										   // accumulation production
	ofstream outfile("Intrinsic_permeability_amgx.txt", ios::app); // output permeability;

	Flag_eigen = false;
	Flag_intri_per = true;
	memory();
	refer_pressure = 0;

	Paramentinput();

	initial_condition();
	para_cal(1);
	Matrix_permeability(1);
	// CSR2COO();

	// begin AMGX initialization
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "solver.json");

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
	AMGX_solver_subroutine_per(A_amgx, b_amgx, solution_amgx, solver, n_amgx, nnz_amgx);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	double end = omp_get_wtime();
	outfile << "permeability = " << (macro + micro_advec) * visco(inlet_pre + refer_pressure, compre(inlet_pre + refer_pressure), Temperature) * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) / 1e-21 << " nD" << "\t"
			<< "pressure = " << inlet_pre + refer_pressure << " Pa" << "\t"
			<< "solve time = " << end - start << " s" << endl;

	if (Flag_outputvtk)
	{
		output(1, 1);
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

int main(int argc, char **argv)
{
	char *buf;
	buf = get_current_dir_name();
	folderPath.assign(buf);
	cout << folderPath << endl;

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
			string eq_head = "=";
			string dot_head = ",";
			string mao_head = ";";
			string::size_type eq_idx{0};
			string::size_type dot_idx{0};
			string::size_type mao_idx{0};
			vector<int> iputings;

			getline(files, sline);
			getline(files, sline);
			getline(files, sline);
			getline(files, sline);
			getline(files, sline);
			eq_idx = 0;
			dot_idx = 0;
			while ((eq_idx = sline.find(eq_head, eq_idx)) != string::npos && (dot_idx = sline.find(mao_head, dot_idx)) != string::npos)
			{
				istringstream ss(sline.substr(eq_idx + 1, dot_idx - eq_idx - 1));
				int ii;
				ss >> ii;
				iputings.push_back(ii);
				eq_idx++;
				dot_idx++;
			}
			Mode = iputings[0];
		}
	}

	PNMsolver Berea;
	// Berea.mean_pore_size();
	switch (Mode)
	{
	case 1:
		Berea.AMGXsolver();
		break;
	case 2:
		Berea.AMGX_permeability_solver(1);
		break;
	case 3:
		Berea.AMGX_permeability_solver();
		break;
	case 4:
		Berea.AMGX_solver_REV();
		break;
	case 5:
		Berea.AMGX_solver_apparent_permeability_REV();
		break;
	case 6:
		Berea.Eigen_solver_per(1); // 1 代表 本征渗透率 计算 没有参数代表 表观渗透率计算
		break;
	case 7:
		Berea.Eigen_solver_per(); // 1 代表 本征渗透率 计算 没有参数代表 表观渗透率计算
		break;
	default:
		break;
	}

	/*产气模拟*/
	// Berea.Eigen_solver();
	// Berea.AMGXsolver();
	// Berea.AMGX_solver_REV();
	/*渗透率计算*/
	// Berea.Eigen_solver_per(1); // 1 代表 本征渗透率 计算 没有参数代表 表观渗透率计算
	// Berea.AMGX_permeability_solver(1); // 1 代表 本征渗透率 计算 没有参数代表 表观渗透率计算
	// Berea.AMGX_solver_apparent_permeability_REV();
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
	// 	auto r = 0.1e-6;
	// 	auto km = 0.388e-15; //K_Clay_HP=62.96e-15;  0.91e-6  #粘土单元本征渗透率 m2    K_Clay_LP=0.388e-15; 0.1e-6  #粘土单元本征渗透率 m2
	// 	auto z{Berea.compre(pressure)};
	// 	auto vis(Berea.visco(pressure, z, Temperature));
	// 	double rho = 0.016 * pressure / (z * 8.314 * Temperature);
	// 	auto Knusen_number = vis / pressure * sqrt(pi * z * 8.314 * Temperature / (2 * 0.016)) / r;
	// 	double alpha = 1.5272 * 2 / pi * atan(2.5  * pow(Knusen_number, 0.5));
	// 	double beta = 6;
	// 	double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
	// 	auto out = Slip * km ;
	// 	Tij_micro << "Tij= " << out/1e-15 << "\t;Knusen_number= " << Knusen_number << "\t;slip= " << Slip << "\t;pressure= " << pressure / 1e6 << endl;
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