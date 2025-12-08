void LiquidMatrixConfig_PressureOnly() {
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
  for (size_t i = 0; i < NA; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp); i++) {
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
    Wi = Pb[i].C1;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
        if (Flag_velocity_bound == true) {
          /*这里用的是孔内部的流速边界条件(已经弃用了这种方法)*/
          B[i - inlet] += -conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[i].pressure - Pb[Tb[j].ID_2].pressure);
        } else {
          B[i - inlet] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
        }
      } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
        B[i - inlet] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      }
      /*debug*/
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);
    COO_A[i - inlet].row = i - inlet;
    COO_A[i - inlet].col = i - inlet;
    COO_A[i - inlet].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - inlet] + counter1].row = i - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
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
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> Pi, Wi, F;
    reverse_mode<double>*Pjs, *Wjs;
    Pjs = new reverse_mode<double>[Pb[i].full_coord];
    Wjs = new reverse_mode<double>[Pb[i].full_coord];

    Pi = Pb[i].pressure + refer_pressure;
    Wi = Pb[i].C1;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      Pjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      Wjs[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
        if (Flag_velocity_bound == true) {
          B[i - para_macro] += -conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[i].pressure - Pb[Tb[j].ID_2].pressure);
        } else {
          B[i - para_macro] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
        }
      } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
        B[i - para_macro] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      }
      /*debug*/
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    COO_A[i - para_macro].row = i - para_macro;
    COO_A[i - para_macro].col = i - para_macro;
    COO_A[i - para_macro].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        COO_A[op + mp + coolist2[i - para_macro] + counter1].row = i - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
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

  /*B debug*/
  // ofstream B_OUT("B_OUT_kong.txt");
  // for (int i = 0; i < (op + mp); i++) {
  //   B_OUT << B[i] << endl;
  // }
  // B_OUT.close();

  // /*Tb debug*/
  // ofstream Tb_out("Tb_out.txt");
  // // Tb_out << "id" << "\t" << "ID1" << "\t" << "ID2" << "\t" << "pressure_ID1" << "\t" << "pressure_ID2" << "\t" << "conductivity" << "\t" << "flux" << "\t" << "B" << endl;
  // for (int i = 0; i < 2 * tn; i++) {
  //   // Tb_out << "Tb[" << i << "] =" << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Tb[i].Conductivity << "\t"
  //   //        << -Tb[i].Conductivity * (Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) << "\t" << Tb[i].Conductivity * Pb[Tb[i].ID_2].pressure << endl;
  //   Tb_out << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Tb[i].Conductivity << endl;
  // }
  // Tb_out.close();
}