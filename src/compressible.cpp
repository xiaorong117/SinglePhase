#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

// #include<eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/IterativeLinearSolvers>
// #include <omp.h>
#include <chrono>
#include <ctime>

#include "GPU.hpp"
using namespace std;
using namespace std::chrono;

// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
static int icount = 0;

int Flag_gpu = false;
////常量设置
double pi = 3.1415927;
double gas_vis = 1.4e-3;                          //粘度
double porosity = 0.31;                           //孔隙率
double ko = 5e-15;                                //΢微孔达西渗透率
double inlet_pre = 1000;                          //进口压力
double outlet_pre = 0;                            //出口压力
double D = 9e-9;                                  //扩散系数
double Effect_D = 0.05 * D;                       //微孔中的有效扩散系数
double voxel_size = 5e-4;                         //像素尺寸，单位m    5.345e-6
int outlet_element_n[3] = {200, 200, 400};        //模型的在各个方向的像素数量，本模拟所用模型为正方形
int inter_i;
// const int NUM_THREADS=4;

const int pn = 656539;
const int tn = 2330254;
const int inlet = 1, op = 326, outlet = 9, m_inlet = 1239, mp = 651154, m_outlet = 3810;
const int macro_n = inlet + op + outlet;
const int micro_n = m_inlet + mp + m_outlet;
const int para_macro = inlet + outlet + m_inlet;
const int NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

Eigen::SparseMatrix<double, Eigen::RowMajor> A0(op + mp, op + mp);
Eigen::VectorXd B0(op + mp, 1);

double getmax_2(double a, double b) {
  return a > b ? a : b;
}

double getmax_3(double a, double b, double c) {
  double temp = getmax_2(a, b);
  temp = getmax_2(temp, c);
  return temp;
}

struct pore {
  double X;
  double Y;
  double Z;
  double Radiu;
  int Half_coord;
  int half_accum;
  int full_coord;
  int full_accum;
  int type;
  double pressure;
  double pressure_old;
  double volume;
  double porosity;
  double entry_radius;
  double cond;
};

struct throat {
  int ID_1;
  int ID_2;
  int n_direction;
  double Radiu;
  double Length;
  double Conductivity;
  double center_x;
  double center_y;
  double center_z;
};

struct throatmerge {
  int ID_1;
  int ID_2;
  double Radiu;
  double Conductivity;
};

class PNMsolver        //定义类
{
 public:
  double *X, *X_old, *geta_X, *B;
  //求解的时间变量
  int *ia, *ja;
  double* a;
  //申请孔喉的动态存储空间
  pore* Pb;
  throat* Tb_in;
  throatmerge* Tb;

  double error;
  int time_step = 0;
  double time_all = 0;
  double dt = 1e-3;
  double dt2 = 1e-3;        //与dt初值相同，用于输出结果文件
  double Q_outlet_macro;
  double Q_outlet_micro;
  double total_p;        // total gas content in research domian
  double norm_inf = 0;
  double eps = 1e-3;        //set residual for dx

  int iterations_number = 0;
  //double total_p=2.75554e-8;

  void memory();                //动态分配存储器
  void Paramentinput();         //孔喉数据导入函数声明
  void para_cal();              //喉道长度等相关参数计算
  void PressureMatrix();        //压力矩阵
  void EigenSolve();            //非线性求解器

  void GPU_solver();

  void GPU_solver_subroutine(GPUObjects& _obj);

  //void MKLsolve();
  double Nor_inf(double A[]);        //误差
  double macro_outlet_flow();        //出口大孔流量
  double micro_outlet_flow();        //出口微孔流量
  void solver();                     //瞬态扩散迭代求解流程
  void output(int n);                //输出VTK文件
  //double permeability();

  ~PNMsolver()        //析构函数，释放动态存储
  {
    delete[] X, X_old, geta_X, B;
    delete[] ia, ja, a;
    delete[] B, Pb, Tb_in, Tb;
  }
};

void PNMsolver::memory() {

  X = new double[pn];
  X_old = new double[pn];
  geta_X = new double[op + mp];
  B = new double[op + mp];

  ia = new int[op + mp + 1];
  ja = new int[NA];
  a = new double[NA];

  Pb = new pore[pn];
  Tb_in = new throat[2 * tn];
  Tb = new throatmerge[2 * tn];

  for (int i = 0; i < op + mp; i++) {
    B[i] = 0;
  }
  for (int i = 0; i < pn; i++) {
    Pb[i].pressure = inlet_pre;
    Pb[i].pressure_old = inlet_pre;
  }
  for (int i = macro_n - outlet; i < macro_n; i++) {
    Pb[i].pressure = outlet_pre;
    Pb[i].pressure_old = outlet_pre;
  }
  for (int i = pn - m_outlet; i < pn; i++) {
    Pb[i].pressure = outlet_pre;
    Pb[i].pressure_old = outlet_pre;
  }
}

void PNMsolver::Paramentinput() {
  ifstream porefile("location_pb.txt", ios::in);
  if (!porefile.is_open()) {
    cout << " can not open poredate" << endl;
  }
  for (int i = 0; i < pn; i++) {
    porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].porosity >> Pb[i].entry_radius >> Pb[i].full_coord >> Pb[i].full_accum;
  }
  porefile.close();
  // for (int i = 0;i < pn;i++)
  // {
  // 	Pb[i].X = voxel_size * Pb[i].X;
  // 	Pb[i].Y = voxel_size * Pb[i].Y;
  // 	Pb[i].Z = voxel_size * Pb[i].Z;
  // 	Pb[i].Radiu = voxel_size * Pb[i].Radiu;
  // }

  cout << Pb[pn - 1].X << "\t" << Pb[pn - 1].Y << endl;

  ifstream throatfile("connectivity_pt_full.txt", ios::in);
  if (!throatfile.is_open()) {
    cout << " can not open throatfile" << endl;
  }

  for (int i = 0; i < 2 * tn; i++) {
    throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
  }
  throatfile.close();

  // for (int i = 0;i < 2 * tn;i++)
  // {
  // 	if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
  // 	{
  // 		Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu;      //pnm部分为喉道的半径
  // 	}
  // 	else
  // 	{
  // 		Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu;     //Darcy区的为接触面积
  // 	}
  // 	Tb_in[i].Length = voxel_size * Tb_in[i].Length;
  // 	Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
  // 	Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
  // 	Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
  // }

  for (int i = 2 * tn - 1; i < 2 * tn; i++) {
    cout << Tb_in[i].ID_1 << "\t" << Tb_in[i].ID_2 << endl;
  }
}

void PNMsolver::para_cal() {
  //计算孔隙的体积
  for (int i = 0; i < pn; i++) {
    if (Pb[i].type == 0) {
      Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3;        //孔隙网络单元

    } else if (Pb[i].type == 1) {
      Pb[i].volume = pow(Pb[i].Radiu, 3);        //正方形微孔单元
    } else {
      Pb[i].volume = pow(Pb[i].Radiu, 3) / 2;        //2×2×1、1×2×2和2×1×2的微孔网格
    }
  }

  for (int i = 0; i < pn; i++) {
    if (i < macro_n) {
      Pb[i].porosity = 1;
    } else {
      if (Pb[i].porosity == 1)        //基质
      {
        Pb[i].porosity = 0.01;
        Pb[i].cond = Pb[i].porosity * pow(5e-6, 2) / (32 * 4.5 * 4.5);
      } else if (Pb[i].porosity == 2)        //裂缝
      {
        Pb[i].porosity = 0.5;
        Pb[i].cond = Pb[i].porosity * pow(5e-5, 2) / (32 * 1.5 * 1.5);
      }
    }
  };
  //Total gas content
  for (int i = inlet; i < macro_n - outlet; i++) {
    total_p += (inlet_pre - outlet_pre) * Pb[i].volume;
  }
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    total_p += Pb[i].porosity * (inlet_pre - outlet_pre) * Pb[i].volume;
  }
  cout << "total_p = " << total_p << endl;
  //水力传导系数计算
  double temp1 = 0, temp2 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0;        //两点流量计算中的临时存储变量

  for (int i = 0; i < 2 * tn; i++) {
    if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
      if (Tb_in[i].Length <= 0)        //剔除可能存在的负喉道长度
      {
        Tb_in[i].Length = 0.5 * voxel_size;
      }
      Tb_in[i].Conductivity = pi * pow(Tb_in[i].Radiu, 4) / (8 * gas_vis * Tb_in[i].Length);
    } else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n)) {
      temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * gas_vis);
      length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
      }
      temp2 = abs(Pb[Tb_in[i].ID_2].cond * Tb_in[i].Radiu * angle2 / (gas_vis * length2));
      /*if (Pb[Tb_in[i].ID_2].type == 1)
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * Pb[Tb_in[i].ID_2].Radiu * 0.5);
			}
			else
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * voxel_size * 0.5);
			}*/
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
    } else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n)) {
      temp2 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * gas_vis);
      length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
      if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1) {
        angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
      } else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3) {
        angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
      } else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5) {
        angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
      }
      temp1 = abs(Pb[Tb_in[i].ID_1].cond * Tb_in[i].Radiu * angle1 / (gas_vis * length1));
      /*if (Pb[Tb_in[i].ID_2].type == 1)
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * Pb[Tb_in[i].ID_2].Radiu * 0.5);
			}
			else
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * voxel_size * 0.5);
			}*/
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);

    } else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet) {
      Tb_in[i].Length = sqrt(pow(Pb[Tb_in[i].ID_1].X - Pb[Tb_in[i].ID_2].X, 2) + pow(Pb[Tb_in[i].ID_1].Y - Pb[Tb_in[i].ID_2].Y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Pb[Tb_in[i].ID_2].Z, 2));
      Tb_in[i].Conductivity = Pb[Tb_in[i].ID_1].cond * Tb_in[i].Radiu / (gas_vis * Tb_in[i].Length);
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
      temp1 = abs(Pb[Tb_in[i].ID_1].cond * Tb_in[i].Radiu * angle1 / (gas_vis * length1));
      temp2 = abs(Pb[Tb_in[i].ID_2].cond * Tb_in[i].Radiu * angle2 / (gas_vis * length2));
      Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
      //cout << temp1 << "\t" << temp2 <<"\t"<< Tb_in[i].Conductivity << endl;
    }
  }

  //merge throat
  int label = 0;
  Tb[0].ID_1 = Tb_in[0].ID_1;
  Tb[0].ID_2 = Tb_in[0].ID_2;
  Tb[0].Radiu = Tb_in[0].Radiu;
  Tb[0].Conductivity = Tb_in[0].Conductivity;
  for (int i = 1; i < 2 * tn; i++) {
    if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2) {
      Tb[label].Conductivity += Tb_in[i].Conductivity;
    } else {
      label++;
      Tb[label].ID_1 = Tb_in[i].ID_1;
      Tb[label].ID_2 = Tb_in[i].ID_2;
      Tb[label].Radiu = Tb_in[i].Radiu;
      Tb[label].Conductivity = Tb_in[i].Conductivity;
    }
  }
  //full_coord
  for (int i = 0; i < pn; i++) {
    Pb[i].full_coord = 0;
    Pb[i].full_accum = 0;
  }

  for (int i = 0; i <= label; i++) {
    Pb[Tb[i].ID_1].full_coord += 1;
  }

  //full_accum
  Pb[0].full_accum = Pb[0].full_coord;
  for (int i = 1; i < pn; i++) {
    Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
  }
  //参数输出验证
  /*for (int i = 0;i < pn;i++)
	{
		cout << "volume[" << i << "]=" << Pb[i].volume << endl;
	}*/

  for (int i = 0; i < 2 * tn; i++) {
    if (Tb_in[i].Conductivity <= 0) {
      cout << "ChannelLength[" << i << "]=" << Tb_in[i].Length << "\t" << "cond[" << i << "]=" << Tb_in[i].Conductivity << endl;
    }
  }
}

void PNMsolver::PressureMatrix() {
  int num;             //每行第一个非0参数的累计编号
  int num1 = 0;        //矩阵中每行的非0数据数量
  int temp;            //确定对角线前面的数据数量
  int temp1;
  int temp2 = 0;

  ia[0] = 1;
  for (int i = 0; i < NA; i++) {
    ja[i] = 0;
    a[i] = 0;
  }

  /* -------------------------------------------------------------------------------------  */
  /* 大孔组装 */
  /* -------------------------------------------------------------------------------------  */
  for (int i = inlet; i < op + inlet; i++) {
    B[i - inlet] = -(Pb[i].pressure - Pb[i].pressure_old);

    temp = 0, temp1 = 0;
    num = ia[i - inlet];
    //macropore
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if (Tb[j].ID_2 < inlet)        //进口
      {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }
      } else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)        //出口
      {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }
      } else {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - inlet] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }

        if (Tb[j].ID_1 > Tb[j].ID_2) {
          temp++;        //矩阵每行对角线值前面的非0值数量
        }
        num1++;        //除对角线值外矩阵每行非0值数量
      }
    }
    num1 += 1;        //加上对角线的非0值
    /*cout << num1 << "\t" << full_coord[i] << endl;*/
    ia[i - inlet + 1] = num1 + 1;        //前i行累计的非零值数量，其中1为ia[0]的值

    ja[num + temp - 1] = i - inlet + 1;        //第i行对角线的值的位置
    a[num + temp - 1] = 1;                     //主对角线的初始值

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if (Tb[j].ID_2 < inlet) {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
        } else {
          a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
        }
      } else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
        } else {
          a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
        }
      } else {
        if (temp1 < temp)        //下三角
        {
          if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
            a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
            a[num - 1] += dt * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / Pb[Tb[j].ID_1].volume;
          } else {
            a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
            a[num - 1] += dt * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
          }

          if (Tb[j].ID_2 < macro_n) {
            ja[num - 1] = Tb[j].ID_2 - inlet + 1;        //下三角值的列位置
          } else {
            ja[num - 1] = Tb[j].ID_2 - para_macro + 1;        //下三角值的列位置
          }
          num++;
          temp1++;
        } else        //上三角
        {
          if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
            a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线

            a[num] += dt * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / Pb[Tb[j].ID_1].volume;
          } else {
            a[ia[i - inlet] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
            a[num] += dt * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
          }
          if (Tb[j].ID_2 < macro_n) {
            ja[num] = Tb[j].ID_2 - inlet + 1;        //下三角值的列位置
          } else {
            ja[num] = Tb[j].ID_2 - para_macro + 1;        //下三角值的列位置
          }
          num++;
        }
      }
    }
  }

  /* -------------------------------------------------------------------------------------  */
  /* 微孔组装 */
  /* -------------------------------------------------------------------------------------  */
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    B[i - para_macro] = -Pb[i].porosity * (Pb[i].pressure - Pb[i].pressure_old);

    temp = 0, temp1 = 0;
    num = ia[i - para_macro];
    //micropore
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)        //微孔进出口边界
      {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }
      } else if (Tb[j].ID_2 >= pn - m_outlet) {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }
      } else {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_1].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        } else {
          B[Tb[j].ID_1 - para_macro] += -dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
        }

        if (Tb[j].ID_1 > Tb[j].ID_2) {
          temp++;        //矩阵每行对角线值前面的非0值数量
        }
        num1++;        //除对角线值外矩阵每行非0值数量
      }
    }
    num1 += 1;        //加上对角线的非0值
    /*cout << num1 << "\t" << full_coord[i] << endl;*/
    ia[i - para_macro + 1] = num1 + 1;              //前i行累计的非零值数量，其中1为ia[0]的值
    ja[num + temp - 1] = i - para_macro + 1;        //第i行对角线的值的位置
    a[num + temp - 1] = Pb[i].porosity;             //主对角线的初始值

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
        } else {
          a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
        }
      } else if (Tb[j].ID_2 >= pn - m_outlet) {
        if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
          a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
        } else {
          a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
        }
      } else {
        if (temp1 < temp)        //下三角
        {
          if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
            a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线
            a[num - 1] += dt * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / Pb[Tb[j].ID_1].volume;
          } else {
            a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
            a[num - 1] += dt * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
          }

          if (Tb[j].ID_2 < macro_n) {
            ja[num - 1] = Tb[j].ID_2 - inlet + 1;        //下三角值的列位置
          } else {
            ja[num - 1] = Tb[j].ID_2 - para_macro + 1;        //下三角值的列位置
          }
          num++;
          temp1++;
        } else        //上三角
        {
          if (Pb[Tb[j].ID_1].pressure > Pb[Tb[j].ID_2].pressure) {
            a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * (2 * Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;        //对角线

            a[num] += dt * Tb[j].Conductivity * (-Pb[Tb[j].ID_1].pressure) / Pb[Tb[j].ID_1].volume;
          } else {
            a[ia[i - para_macro] + temp - 1] += dt * Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure / Pb[Tb[j].ID_1].volume;        //对角线
            a[num] += dt * Tb[j].Conductivity * (Pb[Tb[j].ID_1].pressure - 2 * Pb[Tb[j].ID_2].pressure) / Pb[Tb[j].ID_1].volume;
          }

          if (Tb[j].ID_2 < macro_n) {
            ja[num] = Tb[j].ID_2 - inlet + 1;        //下三角值的列位置
          } else {
            ja[num] = Tb[j].ID_2 - para_macro + 1;        //下三角值的列位置
          }
          num++;
        }
      }
    }
  }

  if (Flag_gpu == true) {
    for (size_t i = 0; i < op + mp + 1; i++) {
      ia[i] += -1;
    }

    for (size_t i = 0; i < ia[op + mp]; i++) {
      ja[i] += -1;
    }
  }

  // for (int i = 1;i < ia[op+mp];i++)
  // {
  // 	cout  << ja[i] << "\t" << a[i] << endl;
  // }

  /*for (int i = 0;i < pn-inlet-outlet ;i++)
	{
		cout<<"B["<<i<<"]=" << B[i] << endl;
	}*/
}

void PNMsolver::GPU_solver() {
  auto start1 = high_resolution_clock::now();
  double acu_flow_macro, acu_flow_micro;
  Flag_gpu = true;
  int n = 1;                                   //label of output file
  int inter_n;                                 //The interation of outer loop of Newton-raphoon method
  double total_flow = 0;                       //accumulation production
  ofstream outfile("Permeability.txt");        //output permeability;

  memory();
  Paramentinput();
  para_cal();
  PressureMatrix();

  // begin GPU initialization
  GPUObjects _obj;
  _obj = GPU_init(ia, ja, a, B, geta_X, op + mp);
  // end GPU initialization

  // ************ begin AMGX solver ************
  GPU_solver_subroutine(_obj);
  do {
    inter_n = 0;
    inter_i = 0;
    do {
      PressureMatrix();
      GPU_solver_subroutine(_obj);
      // EigenSolve();
      //MKLsolve();
      inter_n++;
      cout << "Inf_norm = " << norm_inf << "\t\t" << "dt = " << dt << "\t\t" << "inner loop times = " << inter_n << "\t\t" << "outer loop times= " << time_step << endl;
      cout << endl;
    } while (norm_inf > eps);

    icount = 0;        //重置计数器

    //update pressure_old for next time step
    for (int i = 0; i < pn; i++) {
      Pb[i].pressure_old = Pb[i].pressure;
    }

    cout << "Q_outlet_macro:" << Q_outlet_macro << "\t" << "Q_outlet_micro:" << Q_outlet_micro << endl;
    //cout << "resi_pt_adve:" << Error(Pressure_error) << endl;
    time_all += dt;
    cout << "the " << time_step << " step completed" << endl;
    cout << "time_step:" << dt << "\t" << "tota_time:" << time_all << endl;
    cout << "outer loop interations = " << inter_n << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    //outfile << time_all << "\t" << Q_outlet_macro << "\t" << Q_outlet_micro << endl;

    acu_flow_macro = 0;
    acu_flow_micro = 0;
    for (int i = inlet; i < macro_n - outlet; i++) {
      acu_flow_macro += (inlet_pre - Pb[i].pressure) * Pb[i].volume;
    }
    for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
      acu_flow_micro += Pb[i].porosity * (inlet_pre - Pb[i].pressure) * Pb[i].volume;
    }
    total_flow = acu_flow_macro + acu_flow_micro;
    Q_outlet_macro = macro_outlet_flow();
    Q_outlet_micro = micro_outlet_flow();
    if (time_step % 20 == 0) {
      outfile << time_all << "\t" << Q_outlet_macro << "\t" << Q_outlet_micro << "\t" << acu_flow_macro << "\t" << acu_flow_micro << endl;

      if (inter_i < 10) {
        dt = dt * 2;
      } else if (inter_i > 25) {
        dt = dt / 2;
      }
    }

    if (total_flow / total_p > 0.1 * n) {
      output(time_step);
      n++;
    }
    time_step++;
  } while (total_flow / total_p < 0.99);
  //total_flow / total_p < 0.99
  //while (time_step<100);  while (Error() > 1e-8);
  outfile.close();
  auto stop1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(stop1 - start1);
  cout << "Time-consuming = " << duration1.count() << " MS" << endl;
  ofstream out("calculate time.txt");
  out << duration1.count();
  out.close();
  /***********************销毁AMGX***************************/
  GPU_release(ia, ja, a, B, geta_X, _obj);
  // ************ end AMGX solver ************
};

void PNMsolver::EigenSolve() {
  cout << "EigenSolve" << endl;
  using namespace Eigen;
  // 	initParallel();
  // 	omp_set_num_threads(NUM_THREADS);
  // 	setNbThreads(NUM_THREADS);

  auto start = high_resolution_clock::now();

  for (int i = 0; i < op + mp; ++i) {
    for (int j = ia[i]; j < ia[i + 1]; j++) {
      A0.coeffRef(i, ja[j - 1] - 1) = a[j - 1];
    }
  }

  for (int i = 0; i < op + mp; i++) {
    B0[i] = B[i];
    /*cout << B0[i] << endl;*/
  }

  BiCGSTAB<SparseMatrix<double, RowMajor>> solver;
  solver.setTolerance(pow(10, -5));

  /*solver.setMaxIterations(3);*/
  // 计算分解
  solver.compute(A0);

  VectorXd x = solver.solve(B0);
  std::cout << "#iterations:     " << solver.iterations() << std::endl;
  iterations_number = solver.iterations();
  std::cout << "estimated error: " << solver.error() << std::endl;
  if (solver.iterations() > inter_i) {
    inter_i = solver.iterations();
  }

  /*for (int i = 0;i <op+mp;i++)
	{
		std::cout << "x["<<i<<"] = " << x[i]<<endl;
	}*/

  //矩阵的无穷阶范数
  //norm_inf = x.lpNorm<Eigen::Infinity>();
  //矩阵的二阶范数
  norm_inf = x.norm();

  //更新应力场
  for (int i = inlet; i < inlet + op; i++) {
    Pb[i].pressure += x[i - inlet];
    //cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
  }
  for (int i = op; i < op + mp; i++) {
    Pb[i + inlet + outlet + m_inlet].pressure += x[i];
  }
  ////outlet部分孔设置为解吸出口
  for (int i = 0; i < inlet; i++) {
    Pb[i].pressure += x[Tb[i].ID_2 - inlet];
  }
  for (int i = macro_n; i < macro_n + m_inlet; i++) {
    Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time-consuming = " << duration.count() << " MS" << endl;
  // endTime = clock();
  // cout<<"solver time:"<<(endTime-startTime)/1000<<endl;
  ////计算残差，确定扩散时步
  //for (int i = 0;i < tn;i++)
  //{
  //	if (Tb[i].ID_2 >= macro_n)
  //	{
  //		Pressure_error[i] = (Tb[i].Radiu * Tb[i].Length) / (Tb[i].Conductivity * abs((Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure)));
  //	}
  //	else
  //	{
  //		Pressure_error[i] = (pi * Tb[i].Radiu * Tb[i].Radiu * Tb[i].Length) / (Tb[i].Conductivity * abs((Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure)));
  //	}
  //	/*cout << Pressure_error[i] << "\t" << Sat_error[i]<<endl;*/
  //}
  /*for (int i = 0;i < pn ;i++)
	{
		cout<<"X["<<i<<"]="<<X[i]<<endl;
	}*/
}

// void PNMsolver::MKLsolve()
// {
// 	MKL_INT n = op + mp;
// 	MKL_INT mtype = 11;
// 	MKL_INT nrhs = 1;
// 	void* pt[64];
// 	MKL_INT iparm[64];
// 	MKL_INT maxfct, mnum, phase, error, msglvl;
// 	double ddum;
// 	MKL_INT idum;
// 	const char* uplo;

// 	for (int i = 0;i < 64;i++)
// 	{
// 		iparm[i] = 0;
// 	}
// 	iparm[0] = 1;         /* No solver default */
//     iparm[1] = 2;         /* Fill-in reordering from METIS */
//     iparm[3] = 0;         /* No iterative-direct algorithm */
//     iparm[4] = 0;         /* No user fill-in reducing permutation */
//     iparm[5] = 0;         /* Write solution into x */
//     iparm[6] = 0;         /* Not in use */
//     iparm[7] = 2;         /* Max numbers of iterative refinement steps */
//     iparm[8] = 0;         /* Not in use */
//     iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
//     iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
//     iparm[11] = 0;        /* Conjugate transposed/transpose solve */
//     iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
//     iparm[13] = 0;        /* Output: Number of perturbed pivots */
//     iparm[14] = 0;        /* Not in use */
//     iparm[15] = 0;        /* Not in use */
//     iparm[16] = 0;        /* Not in use */
//     iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
//     iparm[18] = -1;       /* Output: Mflops for LU factorization */
//     iparm[19] = 0;        /* Output: Numbers of CG Iterations */
// 	maxfct = 1;
// 	mnum = 1;
// 	msglvl = 1;
// 	error = 0;

// 	for (int i = 0;i < 64;i++)
// 	{
// 		pt[i] = 0;
// 	}
// /* -------------------------------------------------------------------- */
// /* .. Reordering and Symbolic Factorization. This step also allocates */
// /* all memory that is necessary for the factorization. */
// /* -------------------------------------------------------------------- */
// 	phase = 11;
// 	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//              &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
// 	if ( error != 0 )
//     {
//         printf ("\nERROR during symbolic factorization: %d", error);
//         exit (1);
//     }
//     // printf ("\nReordering completed ... ");
//     // printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
//     // printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
// /* -------------------------------------------------------------------- */
// /* .. Numerical factorization. */
// /* -------------------------------------------------------------------- */
//     phase = 22;
//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//              &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
//     if ( error != 0 )
//     {
//         printf ("\nERROR during numerical factorization: %d", error);
//         exit (2);
//     }
//     // printf ("\nFactorization completed ... ");
// /* -------------------------------------------------------------------- */
// /* .. Back substitution and iterative refinement. */
// /* -------------------------------------------------------------------- */
//     phase = 33;
// 	iparm[11] = 0;   //Ax=b
// 	uplo = "non-transposed";
// 	//printf ("\n\nSolving %s system...\n", uplo);
//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, B, geta_X, &error);
// 	if ( error != 0 )
// 	{
// 		printf ("\nERROR during solution: %d", error);
// 		exit (3);
// 	}

// 	//printf ("\nThe solution of the system is: ");
// 	// for ( int j = 0; j < n; j++ )
// 	// {
// 	// 	cout<<"x["<<j<<"] = "<<geta_X[j]<<endl;
// 	// 	//printf ("\n x [%d] = % f", j, geta_X[j]);
// 	// }
//     //printf ("\n");
// 	//更新应力场
// 	for (int i = inlet;i < inlet + op;i++)
// 	{
// 		Pb[i].pressure += geta_X[i - inlet];
// 	}
// 	for (int i = op;i < op + mp;i++)
// 	{
// 		Pb[i + inlet + outlet + m_inlet].pressure += geta_X[i];
// 	}
// 	////outlet部分孔设置为解吸出口
// 	for (int i = 0;i < inlet;i++)
// 	{
// 		Pb[i].pressure += geta_X[Tb[i].ID_2 - inlet];
// 	}
// 	for (int i = macro_n;i < macro_n + m_inlet;i++)
// 	{
// 		Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
// 	}
// 	norm_inf=Nor_inf(geta_X);

// /* -------------------------------------------------------------------- */
// /* Release internal memory. */
// /* -------------------------------------------------------------------- */
// 	phase = -1;
//     PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//              &n, &ddum, ia, ja, &idum, &nrhs,
//              iparm, &msglvl, &ddum, &ddum, &error);
// }

double PNMsolver::Nor_inf(double A[]) {
  double temp1;
  temp1 = abs(A[0]);
  for (int i = 1; i < op + mp; i++) {
    if (temp1 < abs(A[i])) {
      temp1 = abs(A[i]);
    }
  }
  return temp1;
}

double PNMsolver::macro_outlet_flow() {
  double Q_outlet = 0;
  for (int i = macro_n - outlet; i < macro_n; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      Q_outlet += abs(dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
    }
  }
  return Q_outlet;
}

double PNMsolver::micro_outlet_flow() {
  double Q_outlet = 0;
  for (int i = pn - m_outlet; i < pn; i++) {
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      Q_outlet += abs(dt * (Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
    }
  }
  return Q_outlet;
}

void PNMsolver::output(int n) {
  ostringstream name;
  name << "filename_" << int(n) << ".vtk";
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
  outfile << "LINES" << "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
  for (int i = 0; i < Pb[pn - 1].full_accum; i++) {
    outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
  }
  //输出孔体信息
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
  //输出编号信息
  outfile << "SCALARS NUMBER double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << i << endl;
  }
  //输出压力场信息
  outfile << "SCALARS Pressure double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].pressure - outlet_pre << endl;
  }

  //输出孔类型信息
  outfile << "SCALARS pb_type double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < pn; i++) {
    outfile << Pb[i].type << endl;
  }
  //输出吼道信息
  outfile << "CELL_DATA" << "\t" << Pb[pn - 1].full_accum << endl;
  outfile << "SCALARS throat_EqRadius double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < Pb[pn - 1].full_accum; i++) {

    outfile << Tb[i].Radiu << "\t";
  }

  outfile << "SCALARS conductivity double 1" << endl;
  outfile << "LOOKUP_TABLE table2" << endl;
  for (int i = 0; i < Pb[pn - 1].full_accum; i++) {

    outfile << Tb[i].Conductivity << "\t";
  }
  outfile.close();
}

void PNMsolver::GPU_solver_subroutine(GPUObjects& _obj) {
  auto start = high_resolution_clock::now();
  GPU_solveX(icount, ia, ja, a, B, geta_X, _obj);
  norm_inf = 0;
  for (size_t i = 0; i < op + mp; i++) {
    norm_inf += geta_X[i] * geta_X[i];
  }
  norm_inf = sqrt(norm_inf);
  cout << "Inf_norm = " << norm_inf << endl;

  //更新应力场
  for (int i = inlet; i < inlet + op; i++) {
    Pb[i].pressure += geta_X[i - inlet];
    //cout<<"pore["<<i<<"]="<<Pb[i].pressure-outlet_pre<<endl;
  }
  for (int i = op; i < op + mp; i++) {
    Pb[i + inlet + outlet + m_inlet].pressure += geta_X[i];
  }
  ////outlet部分孔设置为解吸出口
  for (int i = 0; i < inlet; i++) {
    Pb[i].pressure += geta_X[Tb[i].ID_2 - inlet];
  }
  for (int i = macro_n; i < macro_n + m_inlet; i++) {
    Pb[i].pressure = Pb[Tb[Pb[i].full_accum - Pb[i].full_coord].ID_2].pressure;
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time-consuming = " << duration.count() << " MS" << endl;
  /*-----------------------------边界条件---------------------------------*/
};

void PNMsolver::solver() {
  auto start1 = high_resolution_clock::now();
  double acu_flow_macro, acu_flow_micro;

  int n = 1;                                   //label of output file
  int inter_n;                                 //The interation of outer loop of Newton-raphoon method
  double total_flow = 0;                       //accumulation production
  ofstream outfile("Permeability.txt");        //output permeability;

  memory();
  Paramentinput();
  para_cal();
  do {
    inter_n = 0;
    inter_i = 0;
    do {
      PressureMatrix();
      EigenSolve();
      //MKLsolve();
      inter_n++;
      cout << "Inf_norm = " << norm_inf << "\t\t" << "dt = " << dt << "\t\t" << "inner loop times = " << inter_n << "\t\t" << "outer loop times= " << time_step << endl;
      cout << endl;
    } while (norm_inf > eps);
    // PressureMatrix();
    // EigenSolve();

    //update pressure_old for next time step
    for (int i = 0; i < pn; i++) {
      Pb[i].pressure_old = Pb[i].pressure;
    }

    cout << "Q_outlet_macro:" << Q_outlet_macro << "\t" << "Q_outlet_micro:" << Q_outlet_micro << endl;
    //cout << "resi_pt_adve:" << Error(Pressure_error) << endl;
    time_all += dt;
    cout << "the " << time_step << " step completed" << endl;
    cout << "time_step:" << dt << "\t" << "tota_time:" << time_all << endl;
    cout << "outer loop interations = " << inter_n << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    //outfile << time_all << "\t" << Q_outlet_macro << "\t" << Q_outlet_micro << endl;

    acu_flow_macro = 0;
    acu_flow_micro = 0;
    for (int i = inlet; i < macro_n - outlet; i++) {
      acu_flow_macro += (inlet_pre - Pb[i].pressure) * Pb[i].volume;
    }
    for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
      acu_flow_micro += Pb[i].porosity * (inlet_pre - Pb[i].pressure) * Pb[i].volume;
    }
    total_flow = acu_flow_macro + acu_flow_micro;
    Q_outlet_macro = macro_outlet_flow();
    Q_outlet_micro = micro_outlet_flow();
    if (time_step % 20 == 0) {
      outfile << time_all << "\t" << Q_outlet_macro << "\t" << Q_outlet_micro << "\t" << acu_flow_macro << "\t" << acu_flow_micro << endl;

      if (inter_i < 10) {
        dt = dt * 2;
      } else if (inter_i > 25) {
        dt = dt / 2;
      }
    }

    if (total_flow / total_p > 0.1 * n) {
      output(time_step);
      n++;
    }
    time_step++;
  } while (total_flow / total_p < 0.99);
  //total_flow / total_p < 0.99
  //while (time_step<100);  while (Error() > 1e-8);
  outfile.close();
  auto stop1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(stop1 - start1);
  cout << "Time-consuming = " << duration1.count() << " MS" << endl;
  ofstream out("calculate time.txt");
  out << duration1.count();
  out.close();
}

int main() {
  PNMsolver Berea;
  // Berea.solver();
  Berea.GPU_solver();
  cout << "计算结束" << endl;
  return 0;
}
