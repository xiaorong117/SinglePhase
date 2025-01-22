#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"
#include <ctime>
#include <chrono>

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

using namespace std;
using namespace std::chrono;

// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
double iters_globa{0};
////常量设置
double pi = 3.1415927;
double gas_vis = 2e-5;				 // 粘度
double porosity = 0.1;				 // 孔隙率
double ko = 12.3e-21;				 // 微孔达西渗透率 m^2
double inlet_pre = 1e6;				 // 进口压力 Pa
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

	// 申请孔喉的动态存储空间

	pore *Pb;
	throat *Tb_in;
	throatmerge *Tb;

	double error;
	int time_step = Time_step;
	double time_all = pyhsic_time;
	double dt = 1e-8;
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

	void para_cal();				 // 喉道长度等相关参数计算
	void para_cal_in_newton();		 // 在牛顿迭代中计算 克努森数
	void para_cal(double);			 // 喉道长度等相关参数计算
	void para_cal_in_newton(double); // 在牛顿迭代中计算 克努森数
	double compre(double pressure);	 // 压缩系数
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

	void Matrix_permeability();
	void Matrix_permeability(double);
	void Matrix();
	void CSR2COO();
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

	~PNMsolver() // 析构函数，释放动态存储
	{
		delete[] dX, B;
		delete[] ia, ja, a, irn, jcn;
		delete[] Pb, Tb_in, Tb;
	}
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
	Matrix_permeability(1);

	Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
	Eigen::VectorXd B0(op + mp, 1);
	Eigen_subroutine_per(A0, B0);

	macro = macro_outlet_Q();
	micro_advec = micro_outlet_advec_Q();

	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start1);

	outfile << (macro + micro_advec) * gas_vis * domain_size_cubic * voxel_size / (pow(domain_size_cubic * voxel_size, 2) * (inlet_pre - outlet_pre)) << "\t"
			<< inlet_pre/1e6 << "\t"
			<< duration2.count() / 1000 << "s"
			<< endl;
	output(1, 1);
	outfile.close();
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

	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}
}

void PNMsolver::initial_condition()
{
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
				// throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].Length;
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
	// full_coord
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

	// 参数输出验证
	/*for (int i = 0;i < pn;i++)
	{
		cout << "volume[" << i << "]=" << Pb[i].volume << endl;
	}*/

	/*for (int i = 738117;i < tn;i++)
	{
		cout << "ChannelLength[" << i << "]=" << Tb[i].Length << "\t" << "cond[" << i << "]=" << Tb[i].Conductivity << endl;
	}*/
}

void PNMsolver::para_cal(double mode)
{
	// 计算孔隙的体积
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

	// ofstream outfile("conductivity1.txt");
	// for (size_t i = 0; i < Pb[pn - 1].full_accum; i++)
	// {
	// 	outfile << Tb_in[i].ID_1 << "-" << Tb_in[i].ID_2 << "-" << Tb_in[i].Conductivity << "    " << Tb[i].ID_1 << "-" << Tb[i].ID_2 << "-" << Tb[i].Conductivity << endl;
	// }
	// outfile.close();
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

	ofstream outfile("conductivity2.txt");
	for (size_t i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << Tb_in[i].ID_1 << "-" << Tb_in[i].ID_2 << "-" << Tb_in[i].Conductivity << "    " << Tb[i].ID_1 << "-" << Tb[i].ID_2 << "-" << Tb[i].Conductivity << endl;
	}
	outfile.close();
}

void PNMsolver::PressureMatrix()
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
		B[i - inlet] = -0.016 * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt);
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

		ja[num + temp - 1] = i - inlet + 1;													  // 第i行对角线的值的位置
		a[num + temp - 1] = 0.016 * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre); // 主对角线的初始值

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
		B[i - para_macro] = -0.016 * porosity * Pb[i].volume * (Pb[i].pressure / Pb[i].compre - Pb[i].pressure_old / Pb[i].compre_old) / (8.314 * Temperature * dt) - Pb[i].volume / dt * ((1 - 0.016 * Pb[i].pressure / (Pb[i].compre * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure / (1 + K_langmuir * Pb[i].pressure)) - (1 - 0.016 * Pb[i].pressure_old / (Pb[i].compre_old * 8.314 * Temperature * Rho_ad)) * (n_max_ad * K_langmuir * Pb[i].pressure_old / (1 + K_langmuir * Pb[i].pressure_old)));

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
		a[num + temp - 1] = 0.016 * porosity * Pb[i].volume / (8.314 * Temperature * dt * Pb[i].compre) + Pb[i].volume / dt * (-2 * n_max_ad * 0.016 * K_langmuir * Pb[i].pressure - n_max_ad * 0.016 * pow(K_langmuir * Pb[i].pressure, 2) + n_max_ad * K_langmuir * Pb[i].compre * 8.314 * Temperature * Rho_ad) / (Pb[i].compre * 8.314 * Temperature * Rho_ad * pow((1 + K_langmuir * Pb[i].pressure), 2));
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

	BiCGSTAB<SparseMatrix<double, RowMajor>> solver;
	solver.setTolerance(pow(10, -5));
	solver.setMaxIterations(1000);
	/*solver.setMaxIterations(3);*/
	// 计算分解
	solver.compute(A0);

	VectorXd x = solver.solve(B0);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	iterations_number = solver.iterations();
	std::cout << "estimated error: " << solver.error() << std::endl;

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
				<< endl;
		inlet_pre = i * 1e6;
		outlet_pre = inlet_pre - 100;
		Function_DS(inlet_pre);
		initial_condition();
	}
	outfile.close();
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

void PNMsolver::Matrix_permeability(double mode)
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
};

int main(int argc, char **argv)
{
	argc = 1;
	char *name = "Project";
	argv = &name;

	char *buf;
	buf = get_current_dir_name();
	folderPath.assign(buf);
	cout << folderPath << endl;

	PNMsolver Berea;
	/*产气模拟*/
	Berea.Eigen_solver();
	/*渗透率计算*/
	Berea.Eigen_solver_per();       //表观渗透率
	Berea.Eigen_solver_per(1);	//绝对渗透率
	return 0;
}
