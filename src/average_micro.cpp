#include <iostream>  // 提供了控制台输入输出的功能
#include <fstream>  //提供了文件输入输出的功能
#include <string>
#include<math.h>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include<ctime>
using namespace std;  //行提示编译器期望在此文件中使用 C++ 标准库中的内容

// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);

//常量设置
double pi = 3.1415927;
double gas_vis = 1.4e-3;           //粘度
double porosity = 0.239;             //孔隙率
double ko = 2.33635e-13;                 //微孔达西渗透率
double inlet_pre = 100;            //进口压力
double outlet_pre =  0;             //出口压力
double voxel_size = 6.5e-6;          //像素尺寸，单位m
int outlet_element_n[3] = { 400,400,400 };     //模型的在各个方向的像素数量，本模拟所用模型为正方形
double time_factor = 10;           //时间因子，确定计算的时间步
double tortuosity = 1.75;
double aver_micro_radius = 1.81e-6;

const int pn = 13042;  //孔数量
const int tn = 64846;  //吼道数量
const int inlet = 484, outlet = 415, LP = 3912, MP = 8231;  //进口边界单元，出口边界单元，分割后大孔，小孔数量
const int NA = (tn - inlet - outlet) * 2 + LP + MP;        //求解孔喉矩阵中非零的量的个数
int solve_type = 1;			 								//求解器类型，0是直接求解，1是迭代求解

struct pore
{
	double X;
	double Y;
	double Z;
	double Radiu;
	int Half_coord;
	int Accumulative_half;
	int type;
	double porosity;
	double pressure;
	double volume;
};

struct throat
{
	int ID_1;
	int ID_2;
	double Radiu;
	double Length;
	double Conductivity;
};

class PNMsolver                             //定义类
{
public:
	double** A;                              //二维矩阵，存储newton迭代法中的变量项
	double* B;                               //一维数组
	//申请孔喉的动态存储空间
	pore* Pb;
	throat* Tb;

	double error;
	int time_step = 0;
	double time_all = 0;
	double dt = 1e-8;
	double dt2 = 1e-8;      //与dt初值相同，用于输出结果文件
	int iterations_number = 0;

	void memory();                         //动态分配存储器
	void Paramentinput();                  //孔喉数据导入函数声明
	void para_cal();                       //喉道长度等相关参数计算
	void PressureMatrix();                 //压力矩阵
	void EigenSolve();                     //非线性求解器
	void output();                         //输出VTK文件
	double permeability();

	~PNMsolver()                           //析构函数，释放动态存储
	{
		for (int i = 0; i < NA; i++)      //i += 1
		{
			delete[] A[i];
		}
		delete[]B, Pb, Tb;
	}
};

void PNMsolver::memory()
{
	B = new double[LP + MP];
	Pb = new pore[pn];
	Tb = new throat[tn];
	A = new double* [NA];
	for (int i = 0; i < NA; i++)
	{
		A[i] = new double[3];   //数组的每个元素本身是一个长度为3的数组，实际上创建了一个二维数组
	}

	for (int i = 0; i < LP + MP; i++)
	{
		B[i] = 0;
	}
	for (int i = 0; i < pn; i++)
	{
		Pb[i].pressure = 0;
	}
	for (int i = 0; i < inlet; i++)
	{
		Pb[i].pressure = inlet_pre;
	}
	for (int i = pn - outlet; i < pn; i++)  // < or <= ?
	{
		Pb[i].pressure = outlet_pre;
	}
}

void PNMsolver::Paramentinput()
{
	ifstream porefile("size-4-4-01_Dry-6.5um-porosity-0.239-400-400-400-pore-Z.txt", ios::in);  //ios::in-以输入方式打开文件
	if (!porefile.is_open())
	{
		cout << " can not open poredate" << endl;
	}
	for (int i = 0; i < pn; i++)
	{
		porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].Accumulative_half >> Pb[i].type >> Pb[i].porosity;
	}
	porefile.close();
	for (int i = 0; i < pn; i++)
	{
		Pb[i].X = voxel_size * Pb[i].X;   //提取孔网时voxel_size = 1
		Pb[i].Y = voxel_size * Pb[i].Y;
		Pb[i].Z = voxel_size * Pb[i].Z;
		Pb[i].Radiu = voxel_size * Pb[i].Radiu;
	}

	ifstream throatfile("size-4-4-01_Dry-6.5um-porosity-0.239-400-400-400-throat-Z.txt", ios::in);
	if (!throatfile.is_open())
	{
		cout << " can not open throatfile" << endl;
	}

	for (int i = 0; i < tn; i++)
	{
		throatfile >> Tb[i].ID_1 >> Tb[i].ID_2 >> Tb[i].Radiu >> Tb[i].Length;
	}
	throatfile.close();

	for (int i = 0; i < tn; i++)
	{
		Tb[i].Radiu = voxel_size * Tb[i].Radiu;
		Tb[i].Length = voxel_size * Tb[i].Length - Pb[Tb[i].ID_1].Radiu - Pb[Tb[i].ID_2].Radiu;
		if (Tb[i].Length < voxel_size)
		{
			Tb[i].Length = voxel_size;
		}
	}

	/*for (int i = 0;i < tn;i++)
	{
		cout << Tb[i].ID_1<<"\t"<< Tb[i].ID_2 << endl;
	}*/
}

void PNMsolver::para_cal()
{
	//计算孔隙的体积
	for (int i = 0; i < pn; i++)
	{
		Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;
	}
	//传导率计算
	for (int i = 0; i < tn; i++)
	{                     
		double k1, k2;
		double length1, length2;
		double temp1, temp2, temp3, temp4;
		if (Pb[Tb[i].ID_2].type == 0)
		{
			temp1 = pi * pow(Pb[Tb[i].ID_1].Radiu, 4) / (8 * gas_vis * Pb[Tb[i].ID_1].Radiu);
			temp2 = pi * pow(Tb[i].Radiu, 4) / (8 * gas_vis * Tb[i].Length);
			temp3 = pi * pow(Pb[Tb[i].ID_2].Radiu, 4) / (8 * gas_vis * Pb[Tb[i].ID_2].Radiu);
			Tb[i].Conductivity = temp1 * temp2 * temp3 / (temp1 * temp2 + temp2 * temp3 + temp1 * temp3);
		}
		else if (Pb[Tb[i].ID_1].type == 0 && Pb[Tb[i].ID_2].type == 1)
		{
			length1 = Tb[i].Length * Pb[Tb[i].ID_1].Radiu / (Pb[Tb[i].ID_1].Radiu + Pb[Tb[i].ID_2].Radiu);
			length2 = Tb[i].Length * Pb[Tb[i].ID_2].Radiu / (Pb[Tb[i].ID_1].Radiu + Pb[Tb[i].ID_2].Radiu);

			temp1 = pi * pow(Pb[Tb[i].ID_1].Radiu, 4) / (8 * gas_vis * Pb[Tb[i].ID_1].Radiu);
			temp2 = pi * pow(Tb[i].Radiu, 4) / (8 * gas_vis * length1);
			k2 = Pb[Tb[i].ID_2].porosity * pow(aver_micro_radius, 2) / (32 * tortuosity * tortuosity);
			temp3 = k2 * pi * pow(Tb[i].Radiu, 2) / (gas_vis * length2);
			temp4 = k2 * pi * pow(Pb[Tb[i].ID_2].Radiu, 2) / (gas_vis * Pb[Tb[i].ID_2].Radiu);
			Tb[i].Conductivity = temp1 * temp2 * temp3 * temp4 / (temp2 * temp3 * temp4 + temp1 * temp3 * temp4 + temp1 * temp2 * temp4 + temp1 * temp2 * temp3);
		}
		else
		{
			length1 = Tb[i].Length * Pb[Tb[i].ID_1].Radiu / (Pb[Tb[i].ID_1].Radiu + Pb[Tb[i].ID_2].Radiu);
			length2 = Tb[i].Length * Pb[Tb[i].ID_2].Radiu / (Pb[Tb[i].ID_1].Radiu + Pb[Tb[i].ID_2].Radiu);

			k1 = Pb[Tb[i].ID_1].porosity * pow(aver_micro_radius, 2) / (32 * tortuosity * tortuosity);
			k2 = Pb[Tb[i].ID_2].porosity * pow(aver_micro_radius, 2) / (32 * tortuosity * tortuosity);

			temp1 = k1 * pi * pow(Pb[Tb[i].ID_1].Radiu, 2) / (gas_vis * Pb[Tb[i].ID_1].Radiu);
			temp2 = k1 * pi * pow(Tb[i].Radiu, 2) / (gas_vis * length1);
			temp3 = k2 * pi * pow(Tb[i].Radiu, 2) / (gas_vis * length2);
			temp4 = k2 * pi * pow(Pb[Tb[i].ID_2].Radiu, 2) / (gas_vis * Pb[Tb[i].ID_2].Radiu);
			Tb[i].Conductivity = temp1 * temp2 * temp3 * temp4 / (temp2 * temp3 * temp4 + temp1 * temp3 * temp4 + temp1 * temp2 * temp4 + temp1 * temp2 * temp3);
		}
		/*static int iCount{ 0 };
		ofstream file;
		string filename = "micropermeability";
		file.open(filename + ".txt", ios::app | ios::out);
		if (!file.is_open())
		{
			std::cerr << "cannot open the file";
		}
		file << iCount << "\t" << k1 << "\t" << k2 << "\t" << k3 << endl;
		file.close();
		k1 = 0;
		k2 = 0;
		k3 = 0;
		iCount++;*/
	}
	
	ofstream out("debug.txt");
	for (int i = 0; i < tn; i++)
	{
		out << "ID_1 = " << Tb[i].ID_1 << "\t" << "ID_2 = " << Tb[i].ID_2 << "\t" << Tb[i].Length << "\t" << "cond[" << i << "] = " << Tb[i].Conductivity << endl;
	}
	out.close();

	//参数输出验证
	/*for (int i = 0;i < pn;i++)
	{
		cout << "volume[" << i << "]=" << Pb[i].volume << endl;
	}

	for (int i = 0;i < tn;i++)
	{
		cout << "ChannelLength[" << i << "]=" << Tb[i].Length << "\t" << "cond[" << i << "]=" << Tb[i].Conductivity << endl;
	}*/
}

void PNMsolver::PressureMatrix()
{
	for (int i = 0; i < LP + MP; i++)
	{
		B[i] = 0;
	}

	for (int i = 0; i < NA; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			A[i][j] = 0;
		}
	}

	int num = LP + MP;

	for (int i = inlet; i < pn - outlet; i++)
	{
		A[i - inlet][0] = i - inlet;
		A[i - inlet][1] = i - inlet;
	}
	//矩阵对角线数据
	//根据ID_1进行组装
	for (int i = 0; i < tn; i++)
	{
		if (Tb[i].ID_1 >= inlet)         //主对角线
		{
			A[Tb[i].ID_1 - inlet][2] += Tb[i].Conductivity;
		}
		else                             //组装b
		{
			B[Tb[i].ID_2 - inlet] = Tb[i].Conductivity * Pb[Tb[i].ID_1].pressure;
		}

		if (Tb[i].ID_2 < pn - outlet)         //主对角线
		{
			A[Tb[i].ID_2 - inlet][2] += Tb[i].Conductivity;
		}
		else                            //组装b
		{
			B[Tb[i].ID_1 - inlet] = Tb[i].Conductivity * Pb[Tb[i].ID_2].pressure;
		}
	}


	//上下三角组装
	for (int i = 0; i < tn; i++)
	{
		if (Tb[i].ID_1 >= inlet && Tb[i].ID_2 < pn - outlet)      //上三角
		{
			A[num][0] = Tb[i].ID_1 - inlet;
			A[num][1] = Tb[i].ID_2 - inlet;
			A[num][2] = -Tb[i].Conductivity;
			num++;
		}
	}

	for (int i = 0; i < tn; i++)
	{
		if (Tb[i].ID_1 >= inlet && Tb[i].ID_2 < pn - outlet)      //下三角
		{
			A[num][0] = Tb[i].ID_2 - inlet;
			A[num][1] = Tb[i].ID_1 - inlet;
			A[num][2] = -Tb[i].Conductivity;
			num++;
		}
	}

	//for (int i = inlet;i < pn-outlet;i++)
	//{
	//	for (int j = Pb[i].Accumulative_half - Pb[i].Half_coord;j < Pb[i].Accumulative_half;j++)
	//	{
	//		if (Tb[j].ID_2 < inlet + LP)
	//		{
	//			A[Tb[j].ID_1 - inlet][2] += Tb[j].Conductivity;
	//			A[Tb[j].ID_2 - inlet][2] += Tb[j].Conductivity;
	//		}
	//		else if (Tb[j].ID_2 >= inlet + LP && Tb[j].ID_2 < pn-outlet)
	//		{
	//			A[Tb[j].ID_1 - inlet][2] += Tb[j].Conductivity ;
	//			A[Tb[j].ID_2 - inlet][2] += Tb[j].Conductivity ;
	//		}
	//		else if (Tb[j].ID_2 >= pn - outlet && Tb[j].ID_1 < inlet + LP)
	//		{
	//			A[Tb[j].ID_1 - inlet][2] += Tb[j].Conductivity;
	//			B[Tb[j].ID_1 - inlet] = Tb[j].Conductivity* Pb[Tb[j].ID_2].pressure;
	//		}
	//		else if (Tb[j].ID_2 >= pn - outlet && Tb[j].ID_1 >= inlet + LP)
	//		{
	//			A[Tb[j].ID_1 - inlet][2] += Tb[j].Conductivity;

	//			B[Tb[j].ID_1 - inlet] = Tb[j].Conductivity*Pb[Tb[j].ID_2].pressure;
	//		}
	//	}
	//}

	//for (int i = 0;i < inlet;i++)
	//{
	//	if (Tb[i].ID_2 < inlet + LP)
	//	{
	//		A[Tb[i].ID_2 - inlet][2] += Tb[i].Conductivity;
	//		B[Tb[i].ID_2 - inlet] = Tb[i].Conductivity * Pb[Tb[i].ID_1].pressure;
	//	}
	//	else
	//	{
	//		A[Tb[i].ID_2 - inlet][2] += Tb[i].Conductivity;
	//		B[Tb[i].ID_2 - inlet] = Tb[i].Conductivity * Pb[Tb[i].ID_1].pressure;
	//	}
	//}

	//////三角矩阵
	//for (int i = inlet;i < pn - outlet;i++)
	//{
	//	for (int j = Pb[i].Accumulative_half - Pb[i].Half_coord;j < Pb[i].Accumulative_half;j++)
	//	{
	//		if (Tb[j].ID_2 < inlet + LP)
	//		{
	//			A[num][0] = Tb[j].ID_1 - inlet;
	//			A[num][1] = Tb[j].ID_2 - inlet;
	//			A[num][2] = -Tb[j].Conductivity;
	//			num++;
	//			A[num][0] = Tb[j].ID_2 - inlet;
	//			A[num][1] = Tb[j].ID_1 - inlet;
	//			A[num][2] = -Tb[j].Conductivity;
	//			num++;
	//		}
	//		else if (Tb[j].ID_2 >= inlet + LP && Tb[j].ID_2 < pn - outlet)
	//		{
	//			A[num][0] = Tb[j].ID_1 - inlet;
	//			A[num][1] = Tb[j].ID_2 - inlet;
	//			A[num][2] = -Tb[j].Conductivity;
	//			num++;
	//			A[num][0] = Tb[j].ID_2 - inlet;
	//			A[num][1] = Tb[j].ID_1 - inlet;
	//			A[num][2] = -Tb[j].Conductivity;
	//			num++;
	//		}
	//	}
	//}

	/*for (int i = 0;i < NA;i++)
	{
		cout <<i<<"\t"<< A[i][0] <<"\t" << A[i][1] << "\t" << A[i][2] << endl;
	}

	for (int i = 0;i < LP+MP ;i++)
	{
		cout<<"B["<<i<<"]=" << B[i] << endl;
	}*/
}

void PNMsolver::EigenSolve()
{
	using namespace Eigen;
	SparseMatrix < double > A0(LP + MP, LP + MP);
	std::vector < Triplet < double > > triplets;

	for (int i = 0; i < NA; ++i)
	{
		if (A[1][2] != 0)
		{
			triplets.emplace_back(A[i][0], A[i][1], A[i][2]);
		}
	}
	// 初始化稀疏矩阵
	A0.setFromTriplets(triplets.begin(), triplets.end());
	/*std::cout << "A0 = \n" << A0 << std::endl;*/

	VectorXd B0(LP + MP, 1);
	for (int i = 0; i < LP + MP; i++)
	{
		B0[i] = B[i];
		/*cout << B0[i] << endl;*/
	}

	if (solve_type == 0)
	{
		SimplicialLLT<SparseMatrix<double>> solver;
		solver.compute(A0);
		if (solver.info() != Success)
		{
			//DECOMPOSE ERROR
		}
		VectorXd x = solver.solve(B0);
		for (int i = inlet; i < pn - outlet; i++)
		{
			Pb[i].pressure = x[i - inlet];
		}
	}
	else if (solve_type == 1)
	{
		BiCGSTAB<SparseMatrix<double> > solver;
		//solver.setMaxIterations(25000);
		solver.setTolerance(pow(10, -9));
		// ����ֽ�
		solver.compute(A0);
		VectorXd x = solver.solve(B0);
		//����Ӧ����
		for (int i = inlet; i < pn - outlet; i++)
		{
			Pb[i].pressure = x[i - inlet];
		}
		std::cout << "#iterations:     " << solver.iterations() << std::endl;
		std::cout << "estimated error: " << solver.error() << std::endl;
	}

	// // 一个BiCGSTAB的实例
	//BiCGSTAB<SparseMatrix<double> > solver;
	///*solver.setMaxIterations(3);*/
	////solver.setMaxIterations(100000);
	//solver.setTolerance(pow(10, -9));
	//// 计算分解
	//solver.compute(A0);
	//VectorXd B0(LP + MP, 1);
	//for (int i = 0; i < LP + MP; i++)
	//{
	//	B0[i] = B[i];
	//	/*cout << B0[i] << endl;*/
	//}
	//VectorXd x = solver.solve(B0);

	///*for (int i = 0;i < op+mp;i++)
	//{
	//	std::cout << "x["<<i<<"] = " << x[i]<<endl;
	//}*/
	//std::cout << "#iterations:     " << solver.iterations() << std::endl;
	//iterations_number = solver.iterations();
	//std::cout << "estimated error: " << solver.error() << std::endl;
	////更新应力场
	//for (int i = inlet; i < pn - outlet; i++)
	//{
	//	Pb[i].pressure = x[i - inlet];
	//}

	//输出渗透率
	ofstream out("Permeability-1.txt");
	out << permeability() << endl;
	out.close();
}

double PNMsolver::permeability()
{
	double TOTAL_FLOW = 0;
	double P = 0;
	for (int i = 0; i < inlet; i++)
	{
		TOTAL_FLOW += (Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) * Tb[i].Conductivity;
	}

	//模型为正方形
	P = TOTAL_FLOW * gas_vis * outlet_element_n[2] / (voxel_size * outlet_element_n[1] * outlet_element_n[0] * (inlet_pre - outlet_pre));

	return P;
}

void PNMsolver::output()
{
	ostringstream name;
	name << "filename_pressure" << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++) {
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	//输出孔喉连接信息
	outfile << "LINES" << "\t" << tn << "\t" << 3 * tn << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	//输出孔体信息
	outfile << "POINT_DATA " << "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].Radiu << "\t";
	}
	outfile << endl;
	//输出编号信息
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	//输出压力场信息
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure << endl;
	}
	//输出孔类型信息
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}
	//输出孔类型信息
	outfile << "SCALARS pb_porosity double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].porosity << endl;
	}
	//输出吼道信息
	outfile << "CELL_DATA" << "\t" << tn << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << Tb[i].Radiu << "\t";
	}
	outfile.close();
}

int main()
{
	PNMsolver Glass;
	Glass.memory();
	Glass.Paramentinput();
	Glass.para_cal();
	Glass.PressureMatrix();
	Glass.EigenSolve();
	Glass.permeability();
	Glass.output();
	cout << "计算结束" << endl;
	return 0;
}
