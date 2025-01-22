#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>
#include <ctime>
#include <sys/types.h>
#include <dirent.h>
#include <filesystem>
#include <unistd.h> // 函数所在头文件
using namespace std;

const double CLOCKS_PER_SECOND = ((clock_t)1000);
std::string folderPath;
double pi = 3.1415927;
double gas_vis = 2e-5;		
double inlet_pre = 100;		
double outlet_pre = 0;		
double inlet_sat = 1;		
double outlet_sat = 0;		
double D = 9e-9;			
double Effect_D = 0.05 * D; 
double particle_size = 2e-6;
double tortuosity = 1.75;
double average_radius = 0.61e-6; 
double voxel_size = 4e-9;		 
double outlet_element_n = 400;	 
double ko = 12.3e-21;
int simulation_type = 1; 

int pn = 1;
int tn = 1;
int inlet = 1, op = 1, outlet = 1, m_inlet = 1, mp = 1, m_outlet = 1;
int macro_n = inlet + op + outlet;
int micro_n = m_inlet + mp + m_outlet;
int NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

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

class PNMsolver 
{
public:
	double *X,  *B, *Pressure_error;

	double **A; 

	int *Half_coord, *Accumulative_half, *pore_type;				
	double *X_cood, *Y_cood, *Z_cood, *pore_EqRadius, *pore_Volume; 
	double *poro, *pct, *element_k;									

	int *ID_1, *ID_2, *n_direction;							  
	double *throat_EqRadius, *center_x, *center_y, *center_z; 
	double *ChannelLength, *Conductivity;					  
	double error;

	int time_step = 0;
	double time_all = 0;
	double dt = 1e-8;
	double dt2 = 1e-8; 
	int iterations_number = 0;

	void memory();			   
	void Poredateinput();	   
	void Throatdateinput();	   
	void para_cal();		   
	void PressureMatrix();	   
	void SaturabilityMatrix(); 
	void EigenSolve();		   
	double Error(double A[]);  
	void solver();			   
	void output();			   
	double permeability();
	double Pe();
	double Averge_Sat(); 
	~PNMsolver() 
	{
		delete[] X, B, Pressure_error;
		for (int i = 0; i < NA; i++)
		{
			delete[] A[i];
		}
		delete[] A;
		delete[] Half_coord, Accumulative_half, pore_type, X_cood, Y_cood, Z_cood, pore_EqRadius, pore_Volume, poro, pct, element_k;
		delete[] ID_1, ID_2, n_direction, throat_EqRadius, ChannelLength, Conductivity;
		delete[] center_x, center_y, center_z;
	}
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
	NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

	if (flag == false)
	{
		cout << "voxel file missed!" << endl;
		abort();
	}

	cout << "pn = " << pn << endl;
	cout << "tn = " << tn << endl;
	cout << "inlet = " << inlet << "; " << "outlet = " << outlet << "; " << "m_inlet = " << m_inlet << "; " << "m_outlet = " << m_outlet << "; " << "op = " << op << "; " << "mp = " << mp << "; " << endl;




	X = new double[pn];
	B = new double[op + mp];
	Pressure_error = new double[tn];
	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}
	for (int i = 0; i < pn; i++)
	{
		X[i] = 0;
	}
	A = new double *[NA];
	for (int i = 0; i < NA; i++)
	{
		A[i] = new double[3];
	}

	Half_coord = new int[pn];
	Accumulative_half = new int[pn];
	pore_type = new int[pn];
	X_cood = new double[pn];
	Y_cood = new double[pn];
	Z_cood = new double[pn];
	pore_EqRadius = new double[pn];
	pore_Volume = new double[pn];
	poro = new double[pn];
	pct = new double[pn];
	element_k = new double[pn];

	ID_1 = new int[tn];
	ID_2 = new int[tn];
	n_direction = new int[tn];
	throat_EqRadius = new double[tn];
	center_x = new double[tn];
	center_y = new double[tn];
	center_z = new double[tn];
	ChannelLength = new double[tn];
	Conductivity = new double[tn];
	for (int i = 0; i < NA; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			A[i][j] = 0;
		}
	}
	for (int i = 0; i < pn; i++)
	{
		X[i] = inlet_pre;
	}
	for (int i = inlet + op; i < macro_n; i++)
	{
		X[i] = outlet_pre;
	}
	for (int i = pn - m_outlet; i < pn; i++)
	{
		X[i] = outlet_pre;
	}
}

void PNMsolver::Poredateinput()
{
	std::vector<std::string> fileList = getFilesInFolder(folderPath);
	bool flag{false};
	for (const auto &file : fileList)
	{
		if (file.find(string("full_pore_re3_z")) != string::npos)
		{
			ifstream porefile(file, ios::in);
			if (porefile.is_open())
			{
				flag = true;
			}
			for (int i = 0; i < pn; i++)
			{
				double waste{0};
				porefile >> X_cood[i] >> Y_cood[i] >> Z_cood[i] >> waste >> pore_EqRadius[i] >> Half_coord[i] >> Accumulative_half[i] >> pore_type[i] >> waste >> waste >> waste >> waste >> element_k[i];
			}
			porefile.close();
			for (int i = 0; i < pn; i++)
			{
				X_cood[i] = voxel_size * X_cood[i];
				Y_cood[i] = voxel_size * Y_cood[i];
				Z_cood[i] = voxel_size * Z_cood[i];
				pore_EqRadius[i] = voxel_size * pore_EqRadius[i];
				pct[i] = average_radius;
			}
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
		if (file.find(string("200_68jxhybrid_throat_z.txt")) != string::npos)
		{
			ifstream throatfile(file, ios::in);
			if (throatfile.is_open())
			{
				flag = true;
			}
			for (int i = 0; i < tn; i++)
			{
				throatfile >> ID_1[i] >> ID_2[i] >> throat_EqRadius[i] >> center_x[i] >> center_y[i] >> center_z[i] >> n_direction[i] >> ChannelLength[i];
			}
			throatfile.close();

			for (int i = 0; i < tn; i++)
			{
				if (ID_2[i] < macro_n)
				{
					throat_EqRadius[i] = voxel_size * throat_EqRadius[i];
				}
				else
				{
					throat_EqRadius[i] = voxel_size * voxel_size * throat_EqRadius[i];
				}
				ChannelLength[i] = voxel_size * ChannelLength[i];
				center_x[i] = voxel_size * center_x[i];
				center_y[i] = voxel_size * center_y[i];
				center_z[i] = voxel_size * center_z[i];
			}
		}
	}
	if (flag == false)
	{
		cout << "throat file missed!" << endl;
		abort();
	}
}

void PNMsolver::para_cal()
{
	for (int i = 0; i < macro_n; i++)
	{
		pore_Volume[i] = 4 * pi * pow(pore_EqRadius[i], 3) / 3;
	}
	for (int i = macro_n; i < pn; i++)
	{
		if (pore_type[i] == 1)
		{
			pore_Volume[i] = pow(pore_EqRadius[i], 3);
		}
		else
		{
			pore_Volume[i] = pow(pore_EqRadius[i], 3) / 2;
		}
	}
	// for (int i = 0; i < pn; i++)
	// {
	// 	element_k[i] = ko;
	// }

	double temp1, temp2, temp3;
	double angle1, angle2;
	double length1, length2; // �������������е���ʱ�洢����
	for (int i = 0; i < tn; i++)
	{
		if (ID_1[i] < macro_n)
		{
			if (ID_2[i] < macro_n)
			{
				if (ChannelLength[i] <= 0) // �޳����ܴ��ڵĸ���������
				{
					ChannelLength[i] = 0.5 * voxel_size;
				}
				temp1 = pi * pow(pore_EqRadius[ID_1[i]], 4) / (8 * gas_vis * pore_EqRadius[ID_1[i]]);
				temp2 = pi * pow(throat_EqRadius[i], 4) / (8 * gas_vis * ChannelLength[i]); // ���-��׼��ˮ������ϵ��
				temp3 = pi * pow(pore_EqRadius[ID_2[i]], 4) / (8 * gas_vis * pore_EqRadius[ID_2[i]]);
				Conductivity[i] = temp1 * temp2 * temp3 / (temp1 * temp2 + temp2 * temp3 + temp1 * temp3);
			}
			else
			{
				temp1 = pi * pow(pore_EqRadius[ID_1[i]], 3) / (8 * gas_vis); // ���-΢�׼��ˮ������ϵ��
				length2 = sqrt(pow(X_cood[ID_2[i]] - center_x[i], 2) + pow(Y_cood[ID_2[i]] - center_y[i], 2) + pow(Z_cood[ID_2[i]] - center_z[i], 2));
				if (n_direction[i] == 0 || n_direction[i] == 1)
				{
					angle2 = (X_cood[ID_2[i]] - center_x[i]) / length2;
				}
				else if (n_direction[i] == 2 || n_direction[i] == 3)
				{
					angle2 = (Y_cood[ID_2[i]] - center_y[i]) / length2;
				}
				else if (n_direction[i] == 4 || n_direction[i] == 5)
				{
					angle2 = (Z_cood[ID_2[i]] - center_z[i]) / length2;
				}
				temp2 = abs(element_k[ID_2[i]] * throat_EqRadius[i] * angle2 / (gas_vis * length2));
				Conductivity[i] = temp1 * temp2 / (temp1 + temp2);
			}
		}
		else
		{
			if (ID_1[i] >= macro_n + m_inlet && ID_2[i] < pn - m_outlet) // ΢����΢�׼��ˮ������ϵ�����������߽磩
			{
				length1 = sqrt(pow(X_cood[ID_1[i]] - center_x[i], 2) + pow(Y_cood[ID_1[i]] - center_y[i], 2) + pow(Z_cood[ID_1[i]] - center_z[i], 2));
				length2 = sqrt(pow(X_cood[ID_2[i]] - center_x[i], 2) + pow(Y_cood[ID_2[i]] - center_y[i], 2) + pow(Z_cood[ID_2[i]] - center_z[i], 2));
				if (n_direction[i] == 0 || n_direction[i] == 1)
				{
					angle1 = (X_cood[ID_1[i]] - center_x[i]) / length1;
					angle2 = (X_cood[ID_2[i]] - center_x[i]) / length2;
				}
				else if (n_direction[i] == 2 || n_direction[i] == 3)
				{
					angle1 = (Y_cood[ID_1[i]] - center_y[i]) / length1;
					angle2 = (Y_cood[ID_2[i]] - center_y[i]) / length2;
				}
				else if (n_direction[i] == 4 || n_direction[i] == 5)
				{
					angle1 = (Z_cood[ID_1[i]] - center_z[i]) / length1;
					angle2 = (Z_cood[ID_2[i]] - center_z[i]) / length2;
				}
				temp1 = abs(element_k[ID_1[i]] * throat_EqRadius[i] * angle1 / (gas_vis * length1));
				temp2 = abs(element_k[ID_2[i]] * throat_EqRadius[i] * angle2 / (gas_vis * length2));
				Conductivity[i] = temp1 * temp2 / (temp1 + temp2);
			}
			else
			{
				Conductivity[i] = element_k[ID_2[i]] * throat_EqRadius[i] / (gas_vis * ChannelLength[i]); // �߽���΢����΢�׼��ˮ������ϵ��
			}
		}
	}
}

void PNMsolver::PressureMatrix()
{
	for (int i = 0; i < mp + op; i++)
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

	int num = op + mp;
	for (int i = inlet; i < op + inlet; i++)
	{
		A[i - inlet][0] = i - inlet;
		A[i - inlet][1] = i - inlet;
	}
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		A[i - inlet - outlet - m_inlet][0] = i - inlet - outlet - m_inlet;
		A[i - inlet - outlet - m_inlet][1] = i - inlet - outlet - m_inlet;
	}

	for (int i = 0; i < macro_n; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_1[j] >= inlet && ID_2[j] < op + inlet)
			{
				A[ID_1[j] - inlet][2] += Conductivity[j];
				A[ID_2[j] - inlet][2] += Conductivity[j];
			}
			else if (ID_2[j] >= macro_n)
			{
				A[ID_1[j] - inlet][2] += Conductivity[j];
				A[ID_2[j] - inlet - outlet - m_inlet][2] += Conductivity[j];
			}
			if (ID_1[j] < inlet)
			{
				A[ID_2[j] - inlet][2] += Conductivity[j];
			}
			if (ID_2[j] >= macro_n - outlet && ID_2[j] < macro_n)
			{
				A[ID_1[j] - inlet][2] += Conductivity[j];
			}
		}
	}

	for (int i = macro_n; i < pn; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_1[j] >= macro_n + m_inlet)
			{
				A[ID_1[j] - inlet - outlet - m_inlet][2] += Conductivity[j];
			}
			if (ID_2[j] < pn - m_outlet)
			{
				A[ID_2[j] - inlet - outlet - m_inlet][2] += Conductivity[j];
			}
		}
	}
	for (int i = inlet; i < op + inlet; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] < op + inlet && ID_2[j] >= inlet) //&&��||���õ��½��һֱ����
			{
				A[num][0] = ID_1[j] - inlet;
				A[num][1] = ID_2[j] - inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
			else if (ID_2[j] >= macro_n)
			{
				A[num][0] = ID_1[j] - inlet;
				A[num][1] = ID_2[j] - inlet - outlet - m_inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
		}
	}

	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] < pn - m_outlet)
			{
				A[num][0] = ID_1[j] - inlet - outlet - m_inlet;
				A[num][1] = ID_2[j] - inlet - outlet - m_inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
		}
	}

	for (int i = inlet; i < op + inlet; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] < op + inlet && ID_2[j] >= inlet) //&&��||���õ��½��һֱ����
			{
				A[num][0] = ID_2[j] - inlet;
				A[num][1] = ID_1[j] - inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
			else if (ID_2[j] >= inlet + op + outlet)
			{
				A[num][0] = ID_2[j] - inlet - outlet - m_inlet;
				A[num][1] = ID_1[j] - inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
		}
	}

	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] < pn - m_outlet)
			{
				A[num][0] = ID_2[j] - inlet - outlet - m_inlet;
				A[num][1] = ID_1[j] - inlet - outlet - m_inlet;
				A[num][2] = -Conductivity[j];
				num++;
			}
		}
	}

	for (int i = 0; i < inlet; i++)
	{
		B[ID_2[i] - inlet] = Conductivity[i] * X[ID_1[i]];
	}

	for (int i = 1; i < inlet + op; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] >= inlet + op && ID_2[j] < inlet + op + outlet)
			{
				B[ID_1[j] - inlet] = Conductivity[j] * X[ID_2[j]];
			}
		}
	}
	for (int i = macro_n; i < macro_n + m_inlet; i++)
	{
		for (int j = Accumulative_half[i] - Half_coord[i]; j < Accumulative_half[i]; j++)
		{
			B[ID_2[j] - inlet - outlet - m_inlet] += Conductivity[j] * X[ID_1[j]];
		}
	}
	for (int i = Accumulative_half[macro_n - 1]; i < Accumulative_half[pn - 1]; i++)
	{
		if (ID_2[i] >= pn - m_outlet)
		{
			B[ID_1[i] - inlet - outlet - m_inlet] += Conductivity[i] * X[ID_2[i]];
		}
	}

	using namespace Eigen;
	SparseMatrix<double> A0(op + mp, op + mp);
	std::vector<Triplet<double>> triplets;

	for (int i = 0; i < NA; ++i)
	{
		if (A[1][2] != 0)
		{
			triplets.emplace_back(A[i][0], A[i][1], A[i][2]);
		}
	}
	A0.setFromTriplets(triplets.begin(), triplets.end());
	BiCGSTAB<SparseMatrix<double>> solver;
	solver.setMaxIterations(8000);
	solver.setTolerance(pow(10, -8));
	solver.compute(A0);
	VectorXd B0(op + mp, 1);
	for (int i = 0; i < op + mp; i++)
	{
		B0[i] = B[i];
		/*cout << B0[i] << endl;*/
	}

	cout << "ok" << endl;
	VectorXd x = solver.solve(B0);

	for (int i = inlet; i < inlet + op; i++)
	{
		X[i] = x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		X[i + inlet + outlet + m_inlet] = x[i];
	}
	std::cout << "Pressure Solver" << endl;
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	iterations_number = solver.iterations();
	std::cout << "estimated error: " << solver.error() << std::endl;
	std::cout << "Pressure calculation completed" << endl;

	ofstream out("3.1-no coarsening-Permeability.txt");
	out << permeability() << endl;
	out.close();

	ostringstream name;
	name << "3.1- no coarsening-filename_" << "pressure" << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << X_cood[i] << "\t" << Y_cood[i] << "\t" << Z_cood[i] << endl;
	}
	outfile << "LINES" << "\t" << tn << "\t" << 3 * tn << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << 2 << "\t" << ID_1[i] << "\t" << ID_2[i] << endl;
	}
	outfile << "POINT_DATA " << "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << pore_EqRadius[i] * 2 << "\t";
		}
		else
		{
			outfile << pore_EqRadius[i] << "\t";
		}
	}
	outfile << endl;
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << X[i] << endl;
	}
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << pore_type[i] << endl;
	}
	outfile << "CELL_DATA" << "\t" << tn << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << throat_EqRadius[i] << "\t";
	}

	outfile << "SCALARS Conductivity double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << Conductivity[i] << "\t";
	}
	outfile.close();
}

void PNMsolver::EigenSolve()
{
	using namespace Eigen;
	SparseMatrix<double> A0(op + mp, op + mp);
	std::vector<Triplet<double>> triplets;

	for (int i = 0; i < NA; ++i)
	{
		if (A[1][2] != 0)
		{
			triplets.emplace_back(A[i][0], A[i][1], A[i][2]);
		}
	}
	A0.setFromTriplets(triplets.begin(), triplets.end());
	/*std::cout << "A0 = \n" << A0 << std::endl;*/
	BiCGSTAB<SparseMatrix<double>> solver;
	solver.setMaxIterations(10000);
	solver.setTolerance(pow(10, -8));
	solver.compute(A0);
	VectorXd B0(op + mp, 1);
	for (int i = 0; i < op + mp; i++)
	{
		B0[i] = B[i];
	}
	VectorXd x = solver.solve(B0);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	iterations_number = solver.iterations();
	std::cout << "estimated error: " << solver.error() << std::endl;
}

void PNMsolver::output()
{
	ostringstream name;
	name << "filename_" << int(time_all / dt2) << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << X_cood[i] << "\t" << Y_cood[i] << "\t" << Z_cood[i] << endl;
	}
	outfile << "LINES" << "\t" << tn << "\t" << 3 * tn << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << 2 << "\t" << ID_1[i] << "\t" << ID_2[i] << endl;
	}
	outfile << "POINT_DATA " << "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << pore_EqRadius[i] * 2 << "\t";
		}
		else
		{
			outfile << pore_EqRadius[i] << "\t";
		}
	}
	outfile << endl;
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << X[i] << endl;
	}
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << pore_type[i] << endl;
	}
	outfile << "CELL_DATA" << "\t" << tn << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < tn; i++)
	{
		outfile << throat_EqRadius[i] << "\t";
	}
	outfile.close();
}

void PNMsolver::solver()
{
	clock_t startTime, endTime;
	startTime = clock();
	memory();
	Poredateinput();
	para_cal();
	PressureMatrix();
	endTime = clock();
	cout << (double)(endTime - startTime) / CLOCKS_PER_SECOND << "Ms" << endl;
	ofstream out("calculate time.txt");
	out << (double)(endTime - startTime) / CLOCKS_PER_SECOND;
	out.close();
}

double PNMsolver::Error(double A[])
{
	double temp1, temp2;
	temp1 = A[0];
	temp2 = 0;
	for (int i = 1; i < tn; i++)
	{
		temp2 = A[i];
		if (temp1 > temp2)
		{
			temp1 = temp2;
		}
	}
	error = temp1;
	return error;
}

double PNMsolver::permeability()
{
	double TOTAL_FLOW = 0;
	double P = 0;
	for (int i = inlet; i < op + inlet; i++)
	{
		for (int j = Accumulative_half[i - 1]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] >= op + inlet && ID_2[j] < macro_n)
			{
				TOTAL_FLOW += (X[ID_1[j]] - X[ID_2[j]]) * Conductivity[j];
			}
		}
	}

	for (int i = macro_n; i < pn - m_outlet; i++)
	{
		for (int j = Accumulative_half[i - 1]; j < Accumulative_half[i]; j++)
		{
			if (ID_2[j] >= pn - m_outlet)
			{
				TOTAL_FLOW += (X[ID_1[j]] - X[ID_2[j]]) * Conductivity[j];
			}
		}
	}
	P = TOTAL_FLOW * gas_vis / (voxel_size * outlet_element_n * (inlet_pre - outlet_pre));
	return P;
}

int main()
{
	char *buf;
	buf = get_current_dir_name();
	folderPath.assign(buf);
	cout << folderPath << endl;
	PNMsolver Berea;
	Berea.solver();
	return 0;
}
