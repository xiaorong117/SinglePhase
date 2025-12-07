#include "CoordinationNumber.hpp"
#include <vector>

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

coolist.resize(op + mp);         // 非进出口全配位数
coolist5.resize(op + mp);        // 只有进出口的全配位数
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
for (int i = inlet; i < op + inlet; i++) {
  int counter{0};
  for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
    if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
      coolist[i - inlet] += 1;
      counter++;
    } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
      coolist[i - inlet] += 1;
      counter++;
    } else if (Tb[j].ID_2 < inlet) {        // 连接的是大孔进口
      coolist5[i - inlet] += 1;
      counter++;
    } else if (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet) {        // 连接的是微孔进口
      coolist5[i - inlet] += 1;
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
  int counter{0};
  for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
    if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
      coolist[i - para_macro] += 1;
      counter++;
    } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (macro_n + m_inlet + mp)) {
      coolist[i - para_macro] += 1;
      counter++;
    } else if (Tb[j].ID_2 < inlet) {        // 连接的是大孔进口
      coolist5[i - para_macro] += 1;
      counter++;
    } else if (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet) {        // 连接的是微孔进口
      coolist5[i - para_macro] += 1;
      counter++;
    } else {
      counter++;
    }
  }
}

coolist2.resize(op + mp);        // 非进出口累计全配位数
coolist6.resize(op + mp);        // 累计的只有进出口的全配位数

if (Flag_vector_data == true) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < coolist2.size(); i++) {
    std::vector<int>::iterator it = coolist.begin() + i;
    std::vector<int>::iterator it_1 = coolist5.begin() + i;
    coolist2[i] = accumulate(coolist.begin(), it, 0);
    coolist6[i] = accumulate(coolist5.begin(), it_1, 0);
  }
  // 输出vector到文件
  const std::string filename = "vector_data.txt";
  const std::string filename_1 = "vector_data_1.txt";
  writeVectorToFile(coolist2, filename);
  writeVectorToFile(coolist6, filename_1);
} else {
  const std::string filename = "vector_data.txt";
  const std::string filename_1 = "vector_data_1.txt";
  // 从文件读取vector
  std::vector<int> newVec = readVectorFromFile(filename);
  std::vector<int> newVec_1 = readVectorFromFile(filename_1);

  coolist2 = newVec;
  coolist6 = newVec_1;
}

if (Flag_species == true) {
  /*我最开始写的kong的transport程序，没有考虑流量边界 */
  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp) * 3];
  B = new double[(op + mp) * 3];

  ia = new int[(op + mp) * 3 + 1];
  ja = new int[NA * 5];

  a = new double[NA * 5];

  COO_A = new Acoo[NA * 5];
} else if (Flag_QIN_trans == true)
/*加上了Qin说的Neumann边界条件之后，变量数量改变了，transport。*/
{
  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp) * 3 + 2];
  B = new double[(op + mp) * 3 + 2];

  int append_nnz = (inlet + m_inlet) * 4 + 2;

  ia = new int[(op + mp) * 3 + 1 + 2];
  ja = new int[NA * 5 + append_nnz];

  a = new double[NA * 5 + append_nnz];

  COO_A = new Acoo[NA * 5 + append_nnz];
} else if (Flag_QIN_Per == true)
/*加上了Qin说的Neumann边界条件之后，变量数量改变了， 求解压力场。*/
{
  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp) + 2];
  B = new double[(op + mp) + 2];

  ia = new int[(op + mp) + 1 + 2];

  int append_nnz = (inlet + m_inlet) * 2 + 2;

  ja = new int[NA + append_nnz];

  a = new double[NA + append_nnz];

  COO_A = new Acoo[NA + append_nnz];
} else {
  /*本征渗透率计算*/
  NA = accumulate(coolist.begin(), coolist.end(), 0) + op + mp;
  coolist.clear();
  dX = new double[(op + mp)];
  B = new double[(op + mp)];

  ia = new int[(op + mp) + 1];
  ja = new int[NA];

  a = new double[NA];

  COO_A = new Acoo[NA];
}