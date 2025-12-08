void JacobianAssembler_LiquidMatrixConfig_2Transport_Neumann() {
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
  int append_nnz = (inlet + m_inlet) * 4 + 2;
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < NA * 5 + append_nnz; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp) * 3 + 2; i++) {
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
      /*这段被注释掉的程序是为了计算渗透率*/
      // if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
      //   B[i - inlet] = 0;
      // } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
      //   B[i - inlet] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      // }
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);
    B[i - inlet] = -F.val();

    COO_A[i - inlet].row = i - inlet;
    COO_A[i - inlet].col = i - inlet;
    COO_A[i - inlet].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        std::size_t index = op + mp + coolist2[i - inlet] + counter1;
        COO_A[index].row = i - inlet;
        COO_A[index].col = Tb[j].ID_2 - inlet;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
      {
        std::size_t index = op + mp + coolist2[i - inlet] + counter1;
        COO_A[index].row = i - inlet;
        COO_A[index].col = Tb[j].ID_2 - para_macro;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        std::size_t index = NA + coolist6[i - inlet];
        if (exists) {
          COO_A[index].col = op + mp + 1;
        } else {
          COO_A[index].col = op + mp;
        }
        COO_A[index].row = i - inlet;
        COO_A[index].val = Pjs[counter].d(0);
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
      /*这段被注释掉的程序是为了计算渗透率*/
      // if (Tb[j].ID_2 < inlet || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {
      //   B[i - para_macro] = 0;
      // } else if ((pn - m_outlet <= Tb[j].ID_2 && Tb[j].ID_2 < pn) || (macro_n - outlet <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n)) {
      //   B[i - para_macro] += conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val() * (Pb[Tb[j].ID_2].pressure + refer_pressure);
      // }
      Tb[j].Conductivity = conductivity_bulk_kong(Pi, Pjs, Wi, Wjs, i, j).val();
    }

    F = func_BULK_PHASE_FLOW_kong(Pi, Pjs, Wi, Wjs, i);
    F.diff(0, 1);

    B[i - para_macro] = -F.val();

    COO_A[i - para_macro].row = i - para_macro;
    COO_A[i - para_macro].col = i - para_macro;
    COO_A[i - para_macro].val = Pi.d(0);

    size_t counter{0};         // 跳过进出口
    size_t counter1{0};        // COOA内存指标
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
      {
        std::size_t index = op + mp + coolist2[i - para_macro] + counter1;
        COO_A[index].row = i - para_macro;
        COO_A[index].col = Tb[j].ID_2 - inlet;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
      {
        std::size_t index = op + mp + coolist2[i - para_macro] + counter1;
        COO_A[index].row = i - para_macro;
        COO_A[index].col = Tb[j].ID_2 - para_macro;
        COO_A[index].val = Pjs[counter].d(0);

        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        std::size_t index = NA + coolist6[i - para_macro];
        if (exists) {
          COO_A[index].col = op + mp + 1;
        } else {
          COO_A[index].col = op + mp;
        }
        COO_A[index].row = i - para_macro;
        COO_A[index].val = Pjs[counter].d(0);
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
/* TRANSPORT EQUATION SOLEVR */
/* -------------------------------------------------------------------------------------
   */

/* -------------------------------------------------------------------------------------
   */
/* 大孔组装 */
/* -------------------------------------------------------------------------------------
   */
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = inlet; i < op + inlet; i++) {
    reverse_mode<double> P1i, P2i, C1i, C2i, F1, F2;
    reverse_mode<double>*P1js, *P2js, *C1js, *C2js;
    P1js = new reverse_mode<double>[Pb[i].full_coord];
    P2js = new reverse_mode<double>[Pb[i].full_coord];
    C1js = new reverse_mode<double>[Pb[i].full_coord];
    C2js = new reverse_mode<double>[Pb[i].full_coord];

    P1i = Pb[i].pressure + refer_pressure;
    P2i = Pb[i].pressure + refer_pressure;
    C1i = Pb[i].C1;
    C2i = Pb[i].C2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      P1js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      P2js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      C1js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      C2js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C2;
      // Tb[j].Conductivity = conductivity_bulk_kong(P1i, P1js, C1i, C1js, i, j).val();
      // Tb[j].Surface_diff_conduc = conductivity_co2_DISPERSION_kong(P1i, P1js, C1i, C1js, i, j).val();
      // Tb[j].Pore_1 = Pb[i].volume * (C1i.val() - Pb[i].C2_old) / dt;
    }

    F1 = func_TRANSPORT_FLOW_kong(P1i, P1js, C1i, C1js, 1, i);
    F2 = func_TRANSPORT_FLOW_kong(P2i, P2js, C2i, C2js, 2, i);
    F1.diff(0, 1);
    F2.diff(0, 1);

    B[i - inlet + op + mp + 2] = -F1.val();
    B[i - inlet + (op + mp) * 2 + 2] = -F2.val();

    std::size_t index_P1i = i - inlet + NA + (inlet + m_inlet) * 2 + 2;
    COO_A[index_P1i].row = i - inlet + op + mp + 2;
    COO_A[index_P1i].col = i - inlet;
    // COO_A[i - inlet + NA].val = 0;
    COO_A[index_P1i].val = P1i.d(0);

    std::size_t index_C1i = i - inlet + 2 * NA + (inlet + m_inlet) * 3 + 2;
    COO_A[index_C1i].row = i - inlet + op + mp + 2;
    COO_A[index_C1i].col = i - inlet + op + mp + 2;
    COO_A[index_C1i].val = C1i.d(0);

    std::size_t index_P2i = i - inlet + 3 * NA + (inlet + m_inlet) * 3 + 2;
    COO_A[index_P2i].row = i - inlet + (op + mp) * 2 + 2;
    COO_A[index_P2i].col = i - inlet;
    // COO_A[index_P2i].val = 0;
    COO_A[index_P2i].val = P2i.d(0);

    std::size_t index_C2i = i - inlet + 4 * NA + (inlet + m_inlet) * 4 + 2;
    COO_A[index_C2i].row = i - inlet + (op + mp) * 2 + 2;
    COO_A[index_C2i].col = i - inlet + (op + mp) * 2 + 2;
    COO_A[index_C2i].val = C2i.d(0);

    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      index_P1i = op + mp + coolist2[i - inlet] + counter1 + 1 * NA + (inlet + m_inlet) * 2 + 2;
      index_C1i = op + mp + coolist2[i - inlet] + counter1 + 2 * NA + (inlet + m_inlet) * 3 + 2;
      index_P2i = op + mp + coolist2[i - inlet] + counter1 + 3 * NA + (inlet + m_inlet) * 3 + 2;
      index_C2i = op + mp + coolist2[i - inlet] + counter1 + 4 * NA + (inlet + m_inlet) * 4 + 2;
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[index_P1i].row = i - inlet + op + mp + 2;
        COO_A[index_P1i].col = Tb[j].ID_2 - inlet;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);

        COO_A[index_C1i].row = i - inlet + op + mp + 2;
        COO_A[index_C1i].col = Tb[j].ID_2 - inlet + op + mp + 2;
        COO_A[index_C1i].val = C1js[counter].d(0);

        COO_A[index_P2i].row = i - inlet + (op + mp) * 2 + 2;
        COO_A[index_P2i].col = Tb[j].ID_2 - inlet;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);

        COO_A[index_C2i].row = i - inlet + (op + mp) * 2 + 2;
        COO_A[index_C2i].col = Tb[j].ID_2 - inlet + (op + mp) * 2 + 2;
        COO_A[index_C2i].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        index_P1i = op + mp + coolist2[i - inlet] + counter1 + NA + (inlet + m_inlet) * 2 + 2;
        COO_A[index_P1i].row = i - inlet + op + mp + 2;
        COO_A[index_P1i].col = Tb[j].ID_2 - para_macro;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);

        COO_A[index_C1i].row = i - inlet + op + mp + 2;
        COO_A[index_C1i].col = Tb[j].ID_2 - para_macro + op + mp + 2;
        COO_A[index_C1i].val = C1js[counter].d(0);

        COO_A[index_P2i].row = i - inlet + (op + mp) * 2 + 2;
        COO_A[index_P2i].col = Tb[j].ID_2 - para_macro;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);

        COO_A[index_C2i].row = i - inlet + (op + mp) * 2 + 2;
        COO_A[index_C2i].col = Tb[j].ID_2 - para_macro + (op + mp) * 2 + 2;
        COO_A[index_C2i].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        index_P1i = NA * 2 + coolist6[i - inlet] + (inlet + m_inlet) * 2 + 2;
        index_P2i = NA * 4 + coolist6[i - inlet] + (inlet + m_inlet) * 3 + 2;
        if (exists) {
          COO_A[index_P1i].col = op + mp + 1;
          COO_A[index_P2i].col = op + mp + 1;
        } else {
          COO_A[index_P1i].col = op + mp;
          COO_A[index_P2i].col = op + mp;
        }
        COO_A[index_P1i].row = i - inlet + op + mp + 2;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);
        COO_A[index_P2i].row = i - inlet + (op + mp) * 2 + 2;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);
        counter++;
      } else {
        counter++;
      }
    }

    delete[] P1js;
    delete[] P2js;
    delete[] C1js;
    delete[] C2js;
  }

/* -------------------------------------------------------------------------------------
 */
/* 微孔组装 */
/* -------------------------------------------------------------------------------------
 */
// micropore
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = macro_n + m_inlet; i < pn - m_outlet; i++) {
    reverse_mode<double> P1i, P2i, C1i, C2i, F1, F2;
    reverse_mode<double>*P1js, *P2js, *C1js, *C2js;
    P1js = new reverse_mode<double>[Pb[i].full_coord];
    P2js = new reverse_mode<double>[Pb[i].full_coord];
    C1js = new reverse_mode<double>[Pb[i].full_coord];
    C2js = new reverse_mode<double>[Pb[i].full_coord];

    P1i = Pb[i].pressure + refer_pressure;
    P2i = Pb[i].pressure + refer_pressure;
    C1i = Pb[i].C1;
    C2i = Pb[i].C2;

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
    {
      P1js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      P2js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].pressure + refer_pressure;
      C1js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C1;
      C2js[j - (Pb[i].full_accum - Pb[i].full_coord)] = Pb[Tb[j].ID_2].C2;
      // Tb[j].Conductivity = conductivity_bulk_kong(P1i, P1js, C1i, C1js, i, j).val();
      // Tb[j].Surface_diff_conduc = conductivity_co2_DISPERSION_kong(P1i, P1js, C1i, C1js, i, j).val();
      // Tb[j].Pore_1 = Pb[i].volume * (C1i.val() - Pb[i].C2_old) / dt;
    }

    F1 = func_TRANSPORT_FLOW_kong(P1i, P1js, C1i, C1js, 1, i);
    F2 = func_TRANSPORT_FLOW_kong(P2i, P2js, C2i, C2js, 2, i);
    F1.diff(0, 1);
    F2.diff(0, 1);

    B[i - para_macro + op + mp + 2] = -F1.val();
    std::size_t index_P1i = i - para_macro + 1 * NA + (inlet + m_inlet) * 2 + 2;
    COO_A[index_P1i].row = i - para_macro + op + mp + 2;
    COO_A[index_P1i].col = i - para_macro;
    // COO_A[index_P1i].val = 0;
    COO_A[index_P1i].val = P1i.d(0);

    std::size_t index_C1i = i - para_macro + 2 * NA + (inlet + m_inlet) * 3 + 2;
    COO_A[index_C1i].row = i - para_macro + op + mp + 2;
    COO_A[index_C1i].col = i - para_macro + op + mp + 2;
    COO_A[index_C1i].val = C1i.d(0);

    std::size_t index_P2i = i - para_macro + 3 * NA + (inlet + m_inlet) * 3 + 2;
    B[i - para_macro + (op + mp) * 2 + 2] = -F2.val();
    COO_A[index_P2i].row = i - para_macro + (op + mp) * 2 + 2;
    COO_A[index_P2i].col = i - para_macro;
    // COO_A[index_P2i].val = 0;
    COO_A[index_P2i].val = P2i.d(0);

    std::size_t index_C2i = i - para_macro + 4 * NA + (inlet + m_inlet) * 4 + 2;
    COO_A[index_C2i].row = i - para_macro + (op + mp) * 2 + 2;
    COO_A[index_C2i].col = i - para_macro + (op + mp) * 2 + 2;
    COO_A[index_C2i].val = C2i.d(0);
    size_t counter{0};
    size_t counter1{0};

    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      index_P1i = op + mp + coolist2[i - para_macro] + counter1 + 1 * NA + (inlet + m_inlet) * 2 + 2;
      index_C1i = op + mp + coolist2[i - para_macro] + counter1 + 2 * NA + (inlet + m_inlet) * 3 + 2;
      index_P2i = op + mp + coolist2[i - para_macro] + counter1 + 3 * NA + (inlet + m_inlet) * 3 + 2;
      index_C2i = op + mp + coolist2[i - para_macro] + counter1 + 4 * NA + (inlet + m_inlet) * 4 + 2;
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[index_P1i].row = i - para_macro + op + mp + 2;
        COO_A[index_P1i].col = Tb[j].ID_2 - inlet;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);

        COO_A[index_C1i].row = i - para_macro + op + mp + 2;
        COO_A[index_C1i].col = Tb[j].ID_2 - inlet + op + mp + 2;
        COO_A[index_C1i].val = C1js[counter].d(0);

        COO_A[index_P2i].row = i - para_macro + (op + mp) * 2 + 2;
        COO_A[index_P2i].col = Tb[j].ID_2 - inlet;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);

        COO_A[index_C2i].row = i - para_macro + (op + mp) * 2 + 2;
        COO_A[index_C2i].col = Tb[j].ID_2 - inlet + (op + mp) * 2 + 2;
        COO_A[index_C2i].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        index_P1i = op + mp + coolist2[i - para_macro] + counter1 + 1 * NA + (inlet + m_inlet) * 2 + 2;
        COO_A[index_P1i].row = i - para_macro + op + mp + 2;
        COO_A[index_P1i].col = Tb[j].ID_2 - para_macro;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);

        COO_A[index_C1i].row = i - para_macro + op + mp + 2;
        COO_A[index_C1i].col = Tb[j].ID_2 - para_macro + op + mp + 2;
        COO_A[index_C1i].val = C1js[counter].d(0);

        COO_A[index_P2i].row = i - para_macro + (op + mp) * 2 + 2;
        COO_A[index_P2i].col = Tb[j].ID_2 - para_macro;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);

        COO_A[index_C2i].row = i - para_macro + (op + mp) * 2 + 2;
        COO_A[index_C2i].col = Tb[j].ID_2 - para_macro + (op + mp) * 2 + 2;
        COO_A[index_C2i].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((Tb[j].ID_2 < inlet) || (macro_n <= Tb[j].ID_2 && Tb[j].ID_2 < macro_n + m_inlet)) {        // 连接的是进口
        bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), Tb[j].ID_2);
        index_P1i = NA * 2 + coolist6[i - para_macro] + (inlet + m_inlet) * 2 + 2;
        index_P2i = NA * 4 + coolist6[i - para_macro] + (inlet + m_inlet) * 3 + 2;
        if (exists) {
          COO_A[index_P1i].col = op + mp + 1;
          COO_A[index_P2i].col = op + mp + 1;
        } else {
          COO_A[index_P1i].col = op + mp;
          COO_A[index_P2i].col = op + mp;
        }
        COO_A[index_P1i].row = i - para_macro + op + mp + 2;
        // COO_A[index_P1i].val = 0;
        COO_A[index_P1i].val = P1js[counter].d(0);
        COO_A[index_P2i].row = i - para_macro + (op + mp) * 2 + 2;
        // COO_A[index_P2i].val = 0;
        COO_A[index_P2i].val = P2js[counter].d(0);
        counter++;
      } else {
        counter++;
      }
    }

    delete[] P1js;
    delete[] P2js;
    delete[] C1js;
    delete[] C2js;
  }

  /* -------------------------------------------------------------------------------------
 */
  /* 补充方程 */
  /* -------------------------------------------------------------------------------------
 */
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
  B[op + mp] = -F1.val();
  B[op + mp + 1] = -F2.val();

  F1.diff(0, 1);
  F2.diff(0, 1);

  std::size_t index_append1 = NA + (inlet + m_inlet);
  COO_A[index_append1].row = op + mp;
  COO_A[index_append1].col = op + mp;
  COO_A[index_append1].val = P1i.d(0);

  std::size_t index_append2 = NA + (inlet + m_inlet) + 1;
  COO_A[index_append2].row = op + mp + 1;
  COO_A[index_append2].col = op + mp + 1;
  COO_A[index_append2].val = P2i.d(0);

  iCounter1 = 0;
  iCounter2 = 0;

  for (int i = 0; i < inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    index_append1 = NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2;
    index_append2 = NA + (inlet + m_inlet) + 2 + iCounter1;
    if (exists) {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[index_append1].row = op + mp + 1;
          COO_A[index_append1].col = Tb[j].ID_2 - inlet;
          COO_A[index_append1].val = Pjs2[iCounter2].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[index_append1].row = op + mp + 1;
          COO_A[index_append1].col = Tb[j].ID_2 - para_macro;
          COO_A[index_append1].val = Pjs2[iCounter2].d(0);
        }
      }
      iCounter2++;
    } else {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[index_append2].row = op + mp;
          COO_A[index_append2].col = Tb[j].ID_2 - inlet;
          COO_A[index_append2].val = Pjs1[iCounter1].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[index_append2].row = op + mp;
          COO_A[index_append2].col = Tb[j].ID_2 - para_macro;
          COO_A[index_append2].val = Pjs1[iCounter1].d(0);
        }
      }
      iCounter1++;
    }
  }

  for (int i = inlet + op + outlet; i < inlet + op + outlet + m_inlet; i++) {
    bool exists = std::binary_search(inlet_boundary.begin(), inlet_boundary.end(), i);
    index_append1 = NA + (inlet + m_inlet) + 2 + (inlet + m_inlet - inlet_boundary.size()) + iCounter2;
    index_append2 = NA + (inlet + m_inlet) + 2 + iCounter1;
    if (exists) {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[index_append1].row = op + mp + 1;
          COO_A[index_append1].col = Tb[j].ID_2 - inlet;
          COO_A[index_append1].val = Pjs2[iCounter2].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[index_append2].row = op + mp + 1;
          COO_A[index_append2].col = Tb[j].ID_2 - para_macro;
          COO_A[index_append2].val = Pjs2[iCounter2].d(0);
        }
      }
      iCounter2++;
    } else {
      for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)        // 找到pjs
      {
        if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet))        // 连接的是大孔
        {
          COO_A[index_append2].row = op + mp;
          COO_A[index_append2].col = Tb[j].ID_2 - inlet;
          COO_A[index_append2].val = Pjs1[iCounter1].d(0);
        } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet))        // 连接的是微孔
        {
          COO_A[index_append2].row = op + mp;
          COO_A[index_append2].col = Tb[j].ID_2 - para_macro;
          COO_A[index_append2].val = Pjs1[iCounter1].d(0);
        }
      }
      iCounter1++;
    }
  }

  // ofstream B_out("B_out.txt");
  // for (std::size_t i = 0; i < (op + mp) * 3 + 2; i++) {
  //   B_out << B[i] << endl;
  // }
  // B_out.close();

  /*Tb debug*/
  // ofstream Tb_out("Tb_out.txt");
  // // Tb_out << "id" << "\t" << "ID1" << "\t" << "ID2" << "\t" << "pressure_ID1" << "\t" << "pressure_ID2" << "\t" << "conductivity" << "\t" << "flux" << "\t" << "B" << endl;
  // for (int i = 0; i < 2 * tn; i++) {
  //   // Tb_out << "Tb[" << i << "] =" << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Pb[Tb[i].ID_1].pressure << "\t" << Tb[i].Conductivity << "\t"
  //   //        << -Tb[i].Conductivity * (Pb[Tb[i].ID_1].pressure - Pb[Tb[i].ID_2].pressure) << "\t" << Tb[i].Conductivity * Pb[Tb[i].ID_2].pressure << endl;
  //   Tb_out << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << "\t" << Tb[i].Conductivity << "\t" << Tb[i].Surface_diff_conduc << "\t" << Tb[i].Pore_1 << endl;
  // }
  // Tb_out.close();

#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("matrix diff = %.16g\n", end - start);
#endif
};