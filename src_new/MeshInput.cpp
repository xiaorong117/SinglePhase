#include "MeshInput.hpp"
#include "Globals.hpp"
#include "Memory.hpp"
#include "SelfDefinedFunctions.hpp"
// 补充的必需标准库头文件
#include <cassert>         // For assert()
#include <cstdlib>         // For abort()
#include <fstream>         // For std::ifstream, std::ios
#include <iostream>        // For std::cout, std::endl, std::cerr
#include <sstream>         // For std::istringstream
#include <string>          // For std::string, std::string::npos
#include <vector>          // For std::vector (虽然在您的代码片段中已使用 using namespace std;, 但显式包含是好习惯)

using namespace std;
using namespace Solver_property;
// 1. 实现 getInstance()
// 使用 C++11 局部静态变量的初始化特性，保证线程安全和懒汉式初始化
MeshInput& MeshInput::getInstance() {
  static MeshInput instance;
  return instance;
}

void MeshInput::loadMeshStructures() {
  // 1. 获取所有单例和指针 (访问已分配的内存和尺寸)
  Memory& mem = Memory::getInstance();
  pore* Pb = mem.get_Pb();
  throat* Tb_in = mem.get_Tb_in();

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
        ofstream inlet_coo("inlet_coo.txt");
        for (int i = 0; i < pn; i++) {
          double waste{0};
          porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> Pb[i].Radiu >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum;        // REV
          if (Pb[i].full_coord == 0) {
            cout << i << endl;
          }
          Pb[i].full_coord_ori = Pb[i].full_coord;
          Pb[i].full_accum_ori = Pb[i].full_accum;
          Pb[i].km = ko;

          // if (i >= macro_n - outlet && i < macro_n) {
          //   inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          // }
          // if (i >= pn - m_outlet && i < pn) {
          //   inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          // }

          if (i < inlet) {
            inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          } else if (i < macro_n + m_inlet && i >= macro_n) {
            inlet_coo << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << "\t" << i << "\t" << Pb[i].Radiu << "\t" << Pb[i].type << endl;
          }
        }
        porefile.close();
        inlet_coo.close();

        // int count = 980;
        // string filename{"filtered_inlet_coo"};
        // ifstream inlet_coo1(filename + ".txt", ios::in);
        // ostringstream name;
        // name << filename << ".vtk";
        // ofstream outfile(name.str().c_str());
        // outfile << "# vtk DataFile Version 2.0" << endl;
        // outfile << "output.vtk" << endl;
        // outfile << "ASCII " << endl;
        // outfile << "DATASET POLYDATA " << endl;
        // outfile << "POINTS " << count << " float" << endl;
        // for (size_t i = 0; i < count; i++) {
        //   double x, y, z, id, r, type;
        //   inlet_coo1 >> x >> y >> z >> id >> r >> type;
        //   outfile << x << " " << y << " " << z << endl;
        // }
        // inlet_coo1.close();
        // inlet_coo1.open(filename + ".txt", ios::in);
        // outfile << "POINT_DATA " << "\t" << count << endl;
        // outfile << "SCALARS size_pb double 1" << endl;
        // outfile << "LOOKUP_TABLE table1" << endl;
        // for (size_t i = 0; i < count; i++) {
        //   double x, y, z, id, r, type;
        //   inlet_coo1 >> x >> y >> z >> id >> r >> type;
        //   outfile << r << endl;
        // }
        // inlet_coo1.close();

        // inlet_coo1.open(filename + ".txt", ios::in);
        // outfile << "SCALARS NUMBER double 1" << endl;
        // outfile << "LOOKUP_TABLE table2" << endl;
        // for (size_t i = 0; i < count; i++) {
        //   double x, y, z, id, r, type;
        //   inlet_coo1 >> x >> y >> z >> id >> r >> type;
        //   outfile << id << endl;
        // }
        // inlet_coo1.close();

        // inlet_coo1.open(filename + ".txt", ios::in);
        // outfile << "SCALARS TYPE double 1" << endl;
        // outfile << "LOOKUP_TABLE table3" << endl;
        // for (size_t i = 0; i < count; i++) {
        //   double x, y, z, id, r, type;
        //   inlet_coo1 >> x >> y >> z >> id >> r >> type;
        //   outfile << type << endl;
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
    if (Flag_Hybrid == true) {
      if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n) {
        Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu;        // pnm部分为喉道的半径
      } else {
        Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu;        // Darcy区的为接触面积
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

MeshInput::MeshInput() {
  this->pn = 0;
  this->tn = 0;
  this->inlet = 0;
  this->outlet = 0;
  this->m_inlet = 0;
  this->m_outlet = 0;
  this->op = 0;
  this->mp = 0;

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
      this->inlet = num;
      num = int(iputings[1]);
      this->outlet = num;
      num = int(iputings[2]);
      this->m_inlet = num;
      num = int(iputings[3]);
      this->m_outlet = num;
      num = int(iputings[4]);
      this->op = num;
      num = int(iputings[5]);
      this->mp = num;
      num = int(iputings[6]);
      this->pn = num;
      num = int(iputings[7]);
      this->tn = num;
    }
  }

  this->macro_n = inlet + op + outlet;
  this->micro_n = m_inlet + mp + m_outlet;
  this->para_macro = inlet + outlet + m_inlet;
  this->NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

  if (flag == false) {
    cout << "voxel file missed!" << endl;
    abort();
  }

  cout << "pn = " << pn << endl;
  cout << "tn = " << tn << endl;
  cout << "inlet = " << inlet << "; " << "outlet = " << outlet << "; " << "m_inlet = " << m_inlet << "; " << "m_outlet = " << m_outlet << "; " << "op = " << op << "; " << "mp = " << mp << "; "
       << endl;
}