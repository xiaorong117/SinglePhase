void JacobianAssembler_LiquidMatrixConfig_PressureOnly_Neumann() {
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
  int append_nnz = (inlet + m_inlet) * 2 + 2;
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < NA + append_nnz; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp) + 2; i++) {
    B[i] = 0;
  }
  // #ifdef _OPENMP
  // #pragma omp parallel for num_threads(int(OMP_PARA))
  // #endif
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
      /*Neumann boundary condition下，求解不可压缩流的压力场*/
      if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
        B[i - inlet] = 0;
      } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
        B[i - inlet] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      }
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
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        if (exists) {
          COO_A[NA + coolist6[i - inlet]].col = op + mp + 1;
        } else {
          COO_A[NA + coolist6[i - inlet]].col = op + mp;
        }
        COO_A[NA + coolist6[i - inlet]].row = i - inlet;
        COO_A[NA + coolist6[i - inlet]].val = Pjs[counter].d(0);
        counter++;
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
  // micropore
  // #ifdef _OPENMP
  // #pragma omp parallel for num_threads(int(OMP_PARA))
  // #endif
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
        B[i - para_macro] = 0;
      } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
        B[i - para_macro] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      }
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
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        if (exists) {
          COO_A[NA + coolist6[i - para_macro]].col = op + mp + 1;
        } else {
          COO_A[NA + coolist6[i - para_macro]].col = op + mp;
        }
        COO_A[NA + coolist6[i - para_macro]].row = i - para_macro;
        COO_A[NA + coolist6[i - para_macro]].val = Pjs[counter].d(0);
        counter++;
      } else {
        counter++;
      }
    }

    delete[] Pjs;
    delete[] Wjs;
  }

  reverse_mode<double> P1i, P2i, F1, F2;
  reverse_mode<double>*Pjs1, *Pjs2;
  Pjs1 = new reverse_mode<double>[inlet + m_inlet - inlet_boundary.size()];
  Pjs2 = new reverse_mode<double>[inlet_boundary.size()];
  std::size_t iCounter1 = 0;
  std::size_t iCounter2 = 0;
  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    if (exists) {
      P2i = Pb[i].pressure + refer_pressure;
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        Pjs2[iCounter2] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      }
      iCounter2++;
    } else {
      P1i = Pb[i].pressure + refer_pressure;
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        Pjs1[iCounter1] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      }
      iCounter1++;
    }
  }

  for (int i = inlet + op + outlet; i < inlet + op + outlet + m_inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    if (exists) {
      P2i = Pb[i].pressure + refer_pressure;
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        Pjs2[iCounter2] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      }
      iCounter2++;
    } else {
      P1i = Pb[i].pressure + refer_pressure;
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        Pjs1[iCounter1] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      }
      iCounter1++;
    }
  }

  F1 = func_append_kong1(P1i, Pjs1);
  F2 = func_append_kong2(P2i, Pjs2);
  B[op + mp] = 1.02 * 0.01 * 0.01 * 0.01 / 60;
  B[op + mp + 1] = 0.18 * 0.01 * 0.01 * 0.01 / 60;

  F1.diff(0, 1);
  F2.diff(0, 1);

  COO_A[NA + (inlet + m_inlet)].row = op + mp;
  COO_A[NA + (inlet + m_inlet)].col = op + mp;
  COO_A[NA + (inlet + m_inlet)].val = P1i.d(0);

  COO_A[NA + (inlet + m_inlet) + 1].row = op + mp + 1;
  COO_A[NA + (inlet + m_inlet) + 1].col = op + mp + 1;
  COO_A[NA + (inlet + m_inlet) + 1].val = P2i.d(0);

  iCounter1 = 0;
  iCounter2 = 0;

  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    if (exists) {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].row = op + mp + 1;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].col = Tb[j].ID_2 - inlet;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].val = Pjs2[iCounter2].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].row = op + mp + 1;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].col = Tb[j].ID_2 - para_macro;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].val = Pjs2[iCounter2].d(0);
        }
      }
      iCounter2++;
    } else {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].row = op + mp;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].col = Tb[j].ID_2 - inlet;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].val = Pjs1[iCounter1].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].row = op + mp;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].col = Tb[j].ID_2 - para_macro;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].val = Pjs1[iCounter1].d(0);
        }
      }
      iCounter1++;
    }
  }

  for (int i = inlet + op + outlet; i < inlet + op + outlet + m_inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    if (exists) {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].row = op + mp + 1;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].col = Tb[j].ID_2 - inlet;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].val = Pjs2[iCounter2].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].row = op + mp + 1;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].col = Tb[j].ID_2 - para_macro;
          COO_A[NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2].val = Pjs2[iCounter2].d(0);
        }
      }
      iCounter2++;
    } else {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].row = op + mp;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].col = Tb[j].ID_2 - inlet;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].val = Pjs1[iCounter1].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].row = op + mp;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].col = Tb[j].ID_2 - para_macro;
          COO_A[NA + (inlet + m_inlet) + 2 + iCounter1].val = Pjs1[iCounter1].d(0);
        }
      }
      iCounter1++;
    }
  }

  /*B debug*/
  ofstream B_OUT("B_OUT_kong.txt");
  for (int i = 0; i < (op + mp) + 2; i++) {
    B_OUT << B[i] << endl;
  }
  B_OUT.close();

  /*Tb debug*/
  ofstream Tb_out("Tb_out.txt");
  // Tb_out << "id" << "\t" << "ID1" << "\t" << "ID2" << "\t" << "pressure_ID1" << "\t" << "pressure_ID2" << "\t" << "conductivity" << "\t" << "flux" << "\t" << "B" << endl;
  for (int i = 0; i < 2 * tn; i++) {
    // Tb_out << "Tb[" << i << "] =" << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Tb[i].Conductivity << "\t"
    //        << -Tb[i].Conductivity * (Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) << "\t" << Tb[i].Conductivity * Pb[Tb[i].ID_2].pressure << endl;
    Tb_out << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Tb[i].Conductivity << endl;
  }
  Tb_out.close();
}