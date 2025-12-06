#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <equation_solve.cpp>        // 假设这个头文件包含了方程求解的相关函数
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
class GasAdsorptionData {
 private:
  std::vector<double> w_values;
  std::vector<double> dvdw_values;
  std::vector<double> log_w_values;
  double Average_pressure = 1e6;        // 平均压力
  double Average_compre = 1;            // 平均压缩系数
  double Average_visco = 1.4e-5;        // 平均粘度
  double refer_pressure = 1e6;          // 参考压力
  double Temperature = 400;             // 温度，单位为K
  double phi = 0.135;                   // 孔隙度
  double tao = 1.5;                     // 迂曲度
  double mean_free_path = 1e-9;         // 平均自由程
  int Flag_intrin = true;

  // 线性插值函数
  double linearInterpolate(double x, double x0, double x1, double y0, double y1) const { return y0 + (x - x0) * (y1 - y0) / (x1 - x0); }

  // 生成GNUPLOT脚本并绘图
  void plotData(const std::vector<double>& x, const std::vector<double>& y, const std::string& title) const {
    // 创建数据文件
    std::ofstream datafile("plot_data.dat");
    for (size_t i = 0; i < x.size(); ++i) {
      datafile << x[i] << " " << y[i] << "\n";
    }
    datafile.close();

    // 创建GNUPLOT脚本
    std::ofstream scriptfile("plot_script.gp");
    scriptfile << "set terminal pngcairo enhanced font 'Arial,12'\n";
    scriptfile << "set output '" << title << ".png'\n";
    scriptfile << "set title '" << title << "'\n";
    scriptfile << "set xlabel 'w'\n";
    scriptfile << "set ylabel 'dv/dw'\n";
    scriptfile << "set grid\n";
    scriptfile << "plot 'plot_data.dat' using 1:2 with lines lw 2 title 'Original'\n";
    scriptfile << "set output\n";
    scriptfile.close();

    // 执行GNUPLOT
    system("gnuplot plot_script.gp");

    std::cout << "图形已保存为 " << title << ".png\n";
  }

 public:
  // 构造函数，从文件读取数据
  GasAdsorptionData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
      // 跳过空行和注释行
      if (line.empty() || line[0] == '#')
        continue;

      std::istringstream iss(line);
      double w, dvdw;
      if (iss >> w >> dvdw) {
        w_values.push_back(w * 1e-9);        // 将w转换为微米
        log_w_values.push_back(std::log10(w));
        // 检查dvdw是否为NaN或无穷大
        if (std::isnan(dvdw) || std::isinf(dvdw)) {
          dvdw = 0.0;        // 将无效值设为0
        }
        dvdw_values.push_back(dvdw);
      }
    }

    if (w_values.empty()) {
      throw std::runtime_error("文件中没有有效数据");
    }

    // 检查数据是否已排序，如果没有则排序
    if (!std::is_sorted(w_values.begin(), w_values.end())) {
      // 创建一个索引向量并根据w值排序
      std::vector<size_t> indices(w_values.size());
      for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

      std::sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) { return w_values[i1] < w_values[i2]; });

      // 重新排列数据
      std::vector<double> sorted_w, sorted_dvdw;
      for (auto i : indices) {
        sorted_w.push_back(w_values[i]);
        sorted_dvdw.push_back(dvdw_values[i]);
      }

      w_values = std::move(sorted_w);
      dvdw_values = std::move(sorted_dvdw);
    }
  }

  // 修改Average_pressure，Average_compre和Temperature的值
  void setParameters(double pressure, double compre, double temp, double visco, double phi_value, double tao_value, double refer_pressure_value, bool Flag_inrtin_l) {
    Average_pressure = pressure;
    Average_compre = compre;
    Temperature = temp;
    Average_visco = visco;
    phi = phi_value;
    tao = tao_value;
    refer_pressure = refer_pressure_value;
    Flag_intrin = Flag_inrtin_l;
  }

  double knusen(double Average_visco, double Average_pressure, double Average_compre, double Temperature, double w) const {
    // 计算平均自由程
    double Kn = Average_visco / Average_pressure * sqrt(3.14 * Average_compre * 8.314 * Temperature / (2 * 0.016)) / w;
    return Kn;
  }

  void mean_free() { mean_free_path = Average_visco / Average_pressure * sqrt(3.14 * Average_compre * 8.314 * Temperature / (2 * 0.016)); }

  double Function_Slip(double knusen) const {
    double alpha_om = 1.358 * 2 / 3.14 * atan(4 * pow(knusen, 0.4));
    double beta_om = 4;
    double Slip_om = (1 + alpha_om * knusen) * (1 + beta_om * knusen / (1 + knusen));
    return Slip_om;
  }

  // 计算f(w)在[w1, w2]区间内的积分
  double integrate(double w1, double w2) const {
    if (w1 >= w2) {
      throw std::invalid_argument("w1必须小于w2");
    }

    // 确保w1和w2在数据范围内
    w1 = std::max(w1, w_values.front());
    w2 = std::min(w2, w_values.back());

    double integral = 0.0;

    // 找到w1和w2对应的索引范围
    auto it1 = std::lower_bound(w_values.begin(), w_values.end(), w1);
    auto it2 = std::lower_bound(w_values.begin(), w_values.end(), w2);

    size_t idx1 = std::distance(w_values.begin(), it1);
    size_t idx2 = std::distance(w_values.begin(), it2);

    // 处理第一个不完整的区间
    if ((w1 > w_values[idx1 - 1]) && (idx1 > 0)) {
      double kn1 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w1);
      double kn2 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w_values[idx1]);
      double Slip_w1 = Function_Slip(kn1);
      double Slip_w2 = Function_Slip(kn2);
      double y1 = get_dvdw(w1) * w1 * w1 * Slip_w1;
      double y2 = dvdw_values[idx1] * w_values[idx1] * w_values[idx1] * Slip_w2;
      integral += (w_values[idx1] - w1) * (y1 + y2) / 2.0;
    }

    // 处理中间的完整区间
    for (size_t i = idx1; i < idx2; ++i) {
      double w_1 = w_values[i];
      double w_2 = w_values[i + 1];
      double kn1 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w_1);
      double kn2 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w_2);
      double Slip_w1 = Function_Slip(kn1);
      double Slip_w2 = Function_Slip(kn2);
      double y1 = dvdw_values[i] * w_1 * w_1 * Slip_w1;
      double y2 = dvdw_values[i + 1] * w_2 * w_2 * Slip_w2;
      integral += (w_2 - w_1) * (y1 + y2) / 2.0;
    }

    // 处理最后一个不完整的区间
    if (w2 < w_values[idx2]) {
      double kn1 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w2);
      double kn2 = knusen(Average_visco, Average_pressure, Average_compre, Temperature, w_values[idx2]);
      double Slip_w1 = Function_Slip(kn1);
      double Slip_w2 = Function_Slip(kn2);
      double y1 = get_dvdw(w2) * w2 * w2 * Slip_w1;
      double y2 = dvdw_values[idx2] * w_values[idx2] * w_values[idx2] * Slip_w2;
      integral -= (w_values[idx2] - w2) * (y1 + y2) / 2.0;
    }
    return integral;
  }

  // 计算f(w)在[w1, w2]区间内的积分
  double integrate_intrin(double w1, double w2) const {
    if (w1 >= w2) {
      throw std::invalid_argument("w1必须小于w2");
    }

    // 确保w1和w2在数据范围内
    w1 = std::max(w1, w_values.front());
    w2 = std::min(w2, w_values.back());

    double integral = 0.0;

    // 找到w1和w2对应的索引范围
    auto it1 = std::lower_bound(w_values.begin(), w_values.end(), w1);
    auto it2 = std::lower_bound(w_values.begin(), w_values.end(), w2);

    size_t idx1 = std::distance(w_values.begin(), it1);
    size_t idx2 = std::distance(w_values.begin(), it2);

    // 处理第一个不完整的区间
    if ((w1 > w_values[idx1 - 1]) && (idx1 > 0)) {
      double y1 = get_dvdw(w1) * w1 * w1;
      double y2 = dvdw_values[idx1] * w_values[idx1] * w_values[idx1];
      integral += (w_values[idx1] - w1) * (y1 + y2) / 2.0;
    }

    // 处理中间的完整区间
    for (size_t i = idx1; i < idx2; ++i) {
      double w_1 = w_values[i];
      double w_2 = w_values[i + 1];
      double y1 = dvdw_values[i] * w_1 * w_1;
      double y2 = dvdw_values[i + 1] * w_2 * w_2;
      integral += (w_2 - w_1) * (y1 + y2) / 2.0;
    }

    // 处理最后一个不完整的区间
    if (w2 < w_values[idx2]) {
      double y1 = get_dvdw(w2) * w2 * w2;
      double y2 = dvdw_values[idx2] * w_values[idx2] * w_values[idx2];
      integral -= (w_values[idx2] - w2) * (y1 + y2) / 2.0;
    }
    return integral;
  }

  // 计算f(w)在[w1, w2]区间内的积分
  double integrate2(double w1, double w2) const {
    if (w1 >= w2) {
      throw std::invalid_argument("w1必须小于w2");
    }

    // 确保w1和w2在数据范围内
    w1 = std::max(w1, w_values.front());
    w2 = std::min(w2, w_values.back());

    double integral = 0.0;

    // 找到w1和w2对应的索引范围
    auto it1 = std::lower_bound(w_values.begin(), w_values.end(), w1);
    auto it2 = std::lower_bound(w_values.begin(), w_values.end(), w2);

    size_t idx1 = std::distance(w_values.begin(), it1);
    size_t idx2 = std::distance(w_values.begin(), it2);

    // 处理第一个不完整的区间
    if ((w1 > w_values[idx1 - 1]) && (idx1 > 0)) {
      double y1 = get_dvdw(w1);
      double y2 = dvdw_values[idx1];
      integral += (w_values[idx1] - w1) * (y1 + y2) / 2.0;
    }

    // 处理中间的完整区间
    for (size_t i = idx1; i < idx2; ++i) {
      double w_1 = w_values[i];
      double w_2 = w_values[i + 1];
      double y1 = dvdw_values[i];
      double y2 = dvdw_values[i + 1];
      integral += (w_2 - w_1) * (y1 + y2) / 2.0;
    }

    // 处理最后一个不完整的区间
    if ((w2 < w_values[idx2])) {
      double y1 = get_dvdw(w2);
      double y2 = dvdw_values[idx2];
      integral -= (w_values[idx2] - w2) * (y1 + y2) / 2.0;
    }

    return integral;
  }

  // 计算每个microporosity的表观渗透率
  double calculatePermeability(double w1, double w2) const {
    double integral = integrate(w1, w2);
    double V = integrate2(w1, w2);
    double K = integral / V / 32.0 * phi / tao;        // 假设常数为32.0
    return K;
  }

  // 计算每个microporosity的本征渗透率
  double calculatePermeability_intrin(double w1, double w2) const {
    double integral = integrate_intrin(w1, w2);
    double V = integrate2(w1, w2);
    double K = integral / V / 32.0 * phi / tao;        // 假设常数为32.0
    return K;
  }
  // 计算每个microporosity的平均表观孔径
  double calculate_mean_poresize(double w1, double w2) {
    mean_free();
    equation_solve solver(mean_free_path);
    double L = solver.solveForL(integrate(w1, w2) / integrate_intrin(w1, w2));
    return L;
  };
  // 计算每个microporosity的平均本征孔径
  double calculate_mean_poresize_intrin(double w1, double w2) {
    double L = sqrt(integrate_intrin(w1, w2) / integrate2(w1, w2));
    return L;
  };
  // 计算每个microporosity的平均表观渗透率和平均孔径
  std::array<double, 2> calculate_Permeability_aver(double w1, double w2) {
    mean_free();
    equation_solve solver(mean_free_path);
    double W_bar = calculate_mean_poresize(w1, w2);
    double integral = solver.f(W_bar);
    double K = integral * calculatePermeability_intrin(w1, w2);        // 假设常数为32.0
    std::array<double, 2> arr = {K, W_bar};
    return arr;
  };
  // 计算每个microporosity的平均本征渗透率和平均孔径
  std::array<double, 2> calculate_Permeability_aver_intri(double w1, double w2) {
    equation_solve solver(mean_free_path);
    double W_bar = calculate_mean_poresize_intrin(w1, w2);
    double integral = pow(W_bar, 2);
    double K = integral / 32.0 * phi / tao;        // 假设常数为32.0
    std::array<double, 2> arr = {K, W_bar};
    return arr;
  };

  double calculate_Ka_from_Wbar(double w1, double w2) {
    // double w_bar_bar = calculate_Permeability_aver_intri(min_w(),w2)[1];
    // double w_bar = calculate_Permeability_aver(min_w(),w2)[1];
    // 2 nm	4 nm	7.59 nm	13.58 nm
    // 4nm	8 nm	16 nm	32 nm
    // double W{6.4e-9};
    // if (w2 <= 5e-9) {
    //     W = 2e-9; // 2 nm
    // } else if (5E-9 < w2 && w2 <= 9e-9) {
    //     W = 4e-9; // 4 nm
    // } else if (9E-9 < w2 && w2 <= 17e-9) {
    //     W = 7.59e-9; // 8 nm
    // } else if (17E-9 < w2 && w2 <= 33e-9) {
    //     W = 13.58e-9; // 16 nm
    // }

    // double ko = pow(W,2) * 0.134 / 1.5 / 32;
    // double pre = Average_pressure;
    // double z = Average_compre;
    // double rho_g = pre * 0.016 / (z * 8.314 * Temperature); // kg/m3
    // double viscos = Average_visco;				// pa.s
    // double Knusen_number = viscos / pre * sqrt(M_PI * z * 8.314 * Temperature
    // / (2 * 0.016)) / (W); double alpha = 1.358 * 2 / M_PI * atan(4 *
    // pow(Knusen_number, 0.4)); double beta = 4; double Slip = (1 + alpha *
    // Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number)); double
    // K_apparent_w =  Slip * ko;

    // return K_apparent_w;
    // double K_apparent = calculate_Permeability_aver(min_w(),w2)[0];

    // std::array<double,2>arr = {K_apparent_w, K_apparent};
    if (Flag_intrin == true) {
      // return calculatePermeability_intrin(min_w(), w2);        // 使用本征渗透率计算
      // double W = 6.3e-9;        //
      // return pow(W, 2) * 0.134 / 1.5 / 32;
      return w2;
    } else {
      return K_apparent(w1,w2);
      // return calculatePermeability(min_w(),w2); // 使用表观渗透率计算
    }
    // return  calculate_Permeability_aver(min_w(),w2)[0];
  }
  double K_apparent(double w1,double w2) {
    double W = w1;
    double ko = 0;
    // if (w2 <= 5e-9) {
    //   int index = int(refer_pressure / 1e6) - 1;
    //   W = 2.32e-9;        // 2 nm
    //   ko = 11E-21;        // 2 nm
    // } else if (5E-9 < w2 && w2 <= 9e-9) {
    //   int index = int(refer_pressure / 1e6) - 1;
    //   W = 4.86e-9;        // 2 nm
    //   ko = 45E-21;
    // } else if (9E-9 < w2 && w2 <= 17e-9) {
    //   int index = int(refer_pressure / 1e6) - 1;
    //   W = 9.42E-9;        // 2 nm
    //   ko = 161E-21;
    // } else if (17E-9 < w2 && w2 <= 33e-9) {
    //   int index = int(refer_pressure / 1e6) - 1;
    //   W = 17.88E-9;        // 2 nm
    //   ko = 515E-21;
    // } else if (33E-9 < w2 && w2 <= 65e-9) {
    //   W = 36.8E-9;        // 2 nm
    //   ko = 2293.81E-21;
    // } else if (34E-9 < w2 && w2 <= 130e-9) {
    //   W = 50E-9;        // 2 nm
    //   ko = 3904.12E-21;
    // }

    // W = 6.2e-9;
    // ko = pow(5.23e-9, 2) * 0.134 / 1.5 / 32;
    // W = w2;
    ko = w2;
    double pre = Average_pressure;
    double z = Average_compre;
    double rho_g = pre * 0.016 / (z * 8.314 * Temperature);        // kg/m3
    double viscos = Average_visco;                                 // pa.s
    double Knusen_number = viscos / pre * sqrt(M_PI * z * 8.314 * Temperature / (2 * 0.016)) / (W);
    double alpha = 1.358 * 2 / M_PI * atan(4 * pow(Knusen_number, 0.4));
    double beta = 4;  
    double Slip = (1 + alpha * Knusen_number) * (1 + beta * Knusen_number / (1 + Knusen_number));
    double K_apparent_w = Slip * ko;
    return K_apparent_w;
  }
  // 获取dv/dw(w)的函数
  double get_dvdw(double w) const {
    // 检查是否在数据范围内
    if (w <= w_values.front())
      return dvdw_values.front();
    if (w >= w_values.back())
      return dvdw_values.back();

    // 找到w所在的区间
    auto it = std::lower_bound(w_values.begin(), w_values.end(), w);
    size_t idx = std::distance(w_values.begin(), it);

    // 线性插值
    return linearInterpolate(w, w_values[idx - 1], w_values[idx], dvdw_values[idx - 1], dvdw_values[idx]);
  }
  // 获取数据点数量
  size_t size() const { return w_values.size(); }

  // 获取最小w值
  double min_w() const { return w_values.front(); }

  // 获取最大w值
  double max_w() const { return w_values.back(); }

  // 绘制原始数据曲线
  void plotOriginalData() const { plotData(log_w_values, dvdw_values, "Original_dVdW_Data"); }
};

// int main() {
//     try {
//         // 读取数据文件
//         GasAdsorptionData data("Pore_size_distribution.txt");

//         std::cout << "成功读取 " << data.size() << " 个数据点\n";
//         std::cout << "w范围: [" << data.min_w() << ", " << data.max_w() <<
//         "]\n";

//         // 绘制原始数据曲线
//         // data.plotOriginalData();

//         // 示例查询
//         // double test_w;
//         // std::cout << "\n输入要查询的w值 (输入q退出): ";
//         // while (std::cin >> test_w) {
//             // double result = data.get_dvdw(test_w);
//         //     std::cout << "dv/dw(" << test_w << ") = " << result <<
//         std::endl;
//         //     std::cout << "输入要查询的w值 (输入q退出): ";
//         // }

//         // 示例积分计算
//         double w1, w2;
//         std::cout << "\n输入积分下限w1和上限w2 (用空格分隔): ";
//         while (std::cin >> w1 >> w2) {
//             try {
//                 double result = data.calculatePermeability_intrin(w1, w2);
//                 double result2 = data.calculatePermeability(w1, w2);
//                 double r = data.calculate_Permeability_aver_intri(w1, w2)[1];
//                 double r2 = data.calculate_Permeability_aver(w1, w2)[1]
//                 std::cout << "∫(从" << w1 << "到" << w2 << ") f(w) dw = "
//                           << result/1e-21 << "nD" << std::endl
//                           << "results2 = " << result2/1e-21 << "nD" <<
//                           std::endl;
//             } catch (const std::exception& e) {
//                 std::cerr << "错误: " << e.what() << std::endl;
//             }
//             std::cout << "\n输入积分下限w1和上限w2 (用空格分隔): ";
//         }

//     } catch (const std::exception& e) {
//         std::cerr << "错误: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }