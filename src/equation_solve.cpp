#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace std;


class equation_solve
{
private:
    double mean_free_path {0}; 
public:
    equation_solve(double mean_free){
        mean_free_path = mean_free; // 初始化平均自由程
    };

    void set_parameter(double mean_free){   
        mean_free_path = mean_free;
    };

    // 定义函数 f(L)
    double f(double L) {
        double term1 = 1.0 + 1.358 * (2.0 / M_PI) * atan(4.0 * pow(mean_free_path / L, 0.4)) * (mean_free_path / L);
        double term2 = (1.0 + 4.0 * (mean_free_path / L) / (1.0 + mean_free_path / L));
        return term1 * term2;
    }

    // 定义 f(L) - C = 0 的方程
    double equation(double L, double C) {
        if (L <= 0)
        {
            L = 10e-9;
        }
        
        return f(L) - C;
    }

    // 定义方程的导数（数值近似）
    double derivative(double L, double C, double h = 1e-15) {
        return (equation(L + h, C) - equation(L - h, C)) / (2.0 * h);
    }

    // 牛顿迭代法求解
    double solveForL(double C, double initialGuess = 1e-9, double tol = 1e-25, int maxIter = 1000) {
        double L = 1e-9;
        for (int i = 0; i < maxIter; ++i) {
            double fVal = equation(L, C);
            double dfVal = derivative(L, C);
            double delta = fVal / dfVal;
            L -= delta;
            // cout << fixed << setprecision(30);
            // cout << "f(L) = " << f(L) << endl;
            // cout << "C ="  << C << endl;
            if (abs(fVal) < tol && L > 0) {
                break;
            }
        }
        return L;
    }
};


// int main() {
//     double C = 23; //4.25e-25
//     // cout << "Enter the constant C (>0): ";
//     // cin >> C;

//     // if (C <= 0) {
//     //     cerr << "Error: C must be greater than 0." << endl;
//     //     return 1;
//     // }
//     equation_solve EQU(8.64E-9);
//     double L = EQU.solveForL(C);
//     cout << fixed << setprecision(30);
//     cout << "The solution L for f(L) = " << C << " is: " << L << endl;
//     cout << "f(L) = " << EQU.f(L) << endl;

//     ofstream file("f(L).txt");
//     for (size_t i = 1; i < 100000; i++)
//     {
//         double L = i * 1e-20;
//         double mean_free_path = 8.64e-9;
//         double term1 = 1.0 + 1.358 * (2.0 / M_PI) * atan(4.0 * pow(mean_free_path / L, 0.4)) * (mean_free_path / L);
//         double term2 = (1.0 + 4.0 * (mean_free_path / L) / (1.0 + mean_free_path / L));
//         file << L << "\t" << term1 * term2<< endl;
//     }
//     file.close();    
//     return 0;
// }