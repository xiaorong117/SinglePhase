/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <assert.h>
#include "bicgstab.hpp"
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char **argv)
{
    const int maxIterations = 100;
    const double tolerance = 0.0000000001;
    if (argc != 1)
    {
        printf("Wrong number of command line arguments. bicgstab_example accepts no arguments.\n");
        return EXIT_FAILURE;
    }
    int m = -1;
    int *h_A_rows = NULL;
    int *h_A_columns = NULL;
    double *h_A_values = NULL;
    const double alpha = 0.75;
    double beta = 0.0;
    make_test_matrix(&m, &h_A_rows, &h_A_columns, &h_A_values);
    double *rhs = (double *)malloc(m * sizeof(double));

    for (size_t i = 0; i < m; i++)
    {
        rhs[i] = 0;
    }

    for (int i = 0; i < m; i++)
    {
        double rsum = 0.0;
        for (int j = h_A_rows[i]; j < h_A_rows[i + 1]; j++)
        {
            rsum += h_A_values[j] * alpha;
        }
        rhs[i] = rsum;
    }

    ofstream out("rhs.txt");
    for (size_t i = 0; i < m; i++)
    {
        out << "rhs[" << i << "]" << " = " << rhs[i] << std::endl;
    }
    out.close();

    printf("Testing BiCGStab\n");
    // API_BICGSTAB(maxIterations, tolerance, h_A_rows, h_A_columns, h_A_values, m, rhs);
    API_UNPBICGSTAB(maxIterations, tolerance, h_A_rows, h_A_columns, h_A_values, m, rhs);

    ofstream solu("solu.txt");
    for (size_t i = 0; i < m; i++)
    {
        solu << "solu[" << i << "]" << " = " << rhs[i] << std::endl;
    }
    solu.close();
}
