void JacobianAssembler_LiquidMatrixConfig_2Transport_Dirichletcpp() {
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
#ifdef _OPENMP
  double start = omp_get_wtime();
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (size_t i = 0; i < NA * 5; i++) {
    COO_A[i].col = 0;
    COO_A[i].row = 0;
    COO_A[i].val = 0;
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(int(OMP_PARA))
#endif
  for (int i = 0; i < (op + mp) * 3; i++) {
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
    int id = omp_get_thread_num();
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
    }

    F1 = func_TRANSPORT_FLOW_kong(P1i, P1js, C1i, C1js, 1, i);
    F2 = func_TRANSPORT_FLOW_kong(P2i, P2js, C2i, C2js, 2, i);
    F1.diff(0, 1);
    F2.diff(0, 1);

    B[i - inlet + op + mp] = -F1.val();
    COO_A[i - inlet + NA].row = i - inlet + op + mp;
    COO_A[i - inlet + NA].col = i - inlet;
    COO_A[i - inlet + NA].val = 0;
    // COO_A[i - inlet + NA].val = P1i.d(0);

    COO_A[i - inlet + 2 * NA].row = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].col = i - inlet + op + mp;
    COO_A[i - inlet + 2 * NA].val = C1i.d(0);

    COO_A[i - inlet + 3 * NA].row = i - inlet + (op + mp) * 2;
    COO_A[i - inlet + 3 * NA].col = i - inlet;
    COO_A[i - inlet + 3 * NA].val = 0;
    // COO_A[i - inlet + 3 * NA].val = P2i.d(0);

    COO_A[i - inlet + 4 * NA].row = i - inlet + (op + mp) * 2;
    COO_A[i - inlet + 4 * NA].col = i - inlet + (op + mp) * 2;
    COO_A[i - inlet + 4 * NA].val = C2i.d(0);
    B[i - inlet + (op + mp) * 2] = -F2.val();

    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = 0;
        // COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = P1js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = C1js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].row = i - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = P2js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].row = i - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].col = Tb[j].ID_2 - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = 0;
        // COO_A[op + mp + coolist2[i - inlet] + counter1 + NA].val = P1js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].row = i - inlet + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 2 * NA].val = C1js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].row = i - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - inlet] + counter1 + 3 * NA].val = P2js[counter].d(0);

        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].row = i - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].col = Tb[j].ID_2 - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - inlet] + counter1 + 4 * NA].val = C2js[counter].d(0);
        counter++;
        counter1++;
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
    }

    F1 = func_TRANSPORT_FLOW_kong(P1i, P1js, C1i, C1js, 1, i);
    F2 = func_TRANSPORT_FLOW_kong(P2i, P2js, C2i, C2js, 2, i);
    F1.diff(0, 1);
    F2.diff(0, 1);

    B[i - para_macro + op + mp] = -F1.val();
    COO_A[i - para_macro + 1 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 1 * NA].col = i - para_macro;
    COO_A[i - para_macro + 1 * NA].val = 0;
    // COO_A[i - para_macro + 1 * NA].val = P1i.d(0);

    COO_A[i - para_macro + 2 * NA].row = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].col = i - para_macro + op + mp;
    COO_A[i - para_macro + 2 * NA].val = C1i.d(0);

    B[i - para_macro + (op + mp) * 2] = -F2.val();
    COO_A[i - para_macro + 3 * NA].row = i - para_macro + (op + mp) * 2;
    COO_A[i - para_macro + 3 * NA].col = i - para_macro;
    COO_A[i - para_macro + 3 * NA].val = 0;
    // COO_A[i - para_macro + 3 * NA].val = P2i.d(0);

    COO_A[i - para_macro + 4 * NA].row = i - para_macro + (op + mp) * 2;
    COO_A[i - para_macro + 4 * NA].col = i - para_macro + (op + mp) * 2;
    COO_A[i - para_macro + 4 * NA].val = C2i.d(0);
    size_t counter{0};
    size_t counter1{0};
    for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++) {
      if ((inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (op + inlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = P1js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - inlet + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = C1js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].row = i - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].col = Tb[j].ID_2 - inlet;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = P2js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].row = i - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].col = Tb[j].ID_2 - inlet + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else if ((macro_n + m_inlet <= Tb[j].ID_2) && Tb[j].ID_2 < (pn - m_outlet)) {
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - para_macro] + counter1 + 1 * NA].val = P1js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].row = i - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].col = Tb[j].ID_2 - para_macro + op + mp;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 2 * NA].val = C1js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].row = i - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].col = Tb[j].ID_2 - para_macro;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = 0;
        // COO_A[op + mp + coolist2[i - para_macro] + counter1 + 3 * NA].val = P2js[counter].d(0);

        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].row = i - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].col = Tb[j].ID_2 - para_macro + (op + mp) * 2;
        COO_A[op + mp + coolist2[i - para_macro] + counter1 + 4 * NA].val = C2js[counter].d(0);
        counter++;
        counter1++;
      } else {
        counter++;
      }
    }

    delete[] P1js;
    delete[] P2js;
    delete[] C1js;
    delete[] C2js;
  }
#ifdef _OPENMP
  double end = omp_get_wtime();
  printf("matrix diff = %.16g\n", end - start);
#endif
  // ofstream COOA_OUT("COOA_ad_unsorted.txt");

  // for (size_t i = 0; i < 4 * NA - coolist2[op]; i++)
  // {
  // 	COOA_OUT << COO_A[i].row << " " << COO_A[i].col << " " << COO_A[i].val
  // << endl;
  // }

  // ofstream B_OUT("B_OUT.txt");

  // for (size_t i = 0; i < 3 * (op + mp); i++) {
  //   B_OUT << B[i] << endl;
  // }
}