#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cmath>

class GasAdsorptionData {
private:
    std::vector<double> w_values;
    std::vector<double> dvdw_values;
    std::vector<double> log_w_values;
    
    // 线性插值函数
    double linearInterpolate(double x, double x0, double x1, double y0, double y1) const {
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    }

    // 生成GNUPLOT脚本并绘图
    void plotData(const std::vector<double>& x, const std::vector<double>& y, 
                 const std::string& title) const {
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
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            double w, dvdw;
            if (iss >> w >> dvdw) {
                w_values.push_back(w);
                log_w_values.push_back(std::log10(w));
                // 检查dvdw是否为NaN或无穷大
                if (std::isnan(dvdw) || std::isinf(dvdw)) {
                    dvdw = 0.0; // 将无效值设为0
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
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
            
            std::sort(indices.begin(), indices.end(), 
                [this](size_t i1, size_t i2) { return w_values[i1] < w_values[i2]; });
            
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
        if (w1 > w_values[idx1-1]) {
            double y1 = get_dvdw(w1);
            double y2 = dvdw_values[idx1];
            integral += (w_values[idx1] - w1) * (y1 + y2) / 2.0;
        }

        // 处理中间的完整区间
        for (size_t i = idx1 + 1; i < idx2; ++i) {
            integral += (w_values[i] - w_values[i-1]) * 
                       (dvdw_values[i-1] + dvdw_values[i]) / 2.0;
        }

        // 处理最后一个不完整的区间
        if (w2 < w_values[idx2]) {
            double y1 = dvdw_values[idx2-1];
            double y2 = get_dvdw(w2);
            integral += (w2 - w_values[idx2-1]) * (y1 + y2) / 2.0;
        }

        return integral;
    }

    // 获取dv/dw(w)的函数
    double get_dvdw(double w) const {
        // 检查是否在数据范围内
        if (w <= w_values.front()) return dvdw_values.front();
        if (w >= w_values.back()) return dvdw_values.back();

        // 找到w所在的区间
        auto it = std::lower_bound(w_values.begin(), w_values.end(), w);
        size_t idx = std::distance(w_values.begin(), it);

        // 线性插值
        return linearInterpolate(w, w_values[idx-1], w_values[idx], 
                               dvdw_values[idx-1], dvdw_values[idx]);
    }

    // 获取数据点数量
    size_t size() const {
        return w_values.size();
    }

    // 获取最小w值
    double min_w() const {
        return w_values.front();
    }

    // 获取最大w值
    double max_w() const {
        return w_values.back();
    }

    // 绘制原始数据曲线
    void plotOriginalData() const {
        plotData(log_w_values, dvdw_values, "Original_dVdW_Data");
    }
};

int main() {
    try {
        // 读取数据文件
        GasAdsorptionData data("Pore_size_distribution.txt");
        
        std::cout << "成功读取 " << data.size() << " 个数据点\n";
        std::cout << "w范围: [" << data.min_w() << ", " << data.max_w() << "]\n";
        
        // 绘制原始数据曲线
        // data.plotOriginalData();
        
        // 示例查询
        // double test_w;
        // std::cout << "\n输入要查询的w值 (输入q退出): ";
        // while (std::cin >> test_w) {
        //     double result = data.get_dvdw(test_w);
        //     std::cout << "dv/dw(" << test_w << ") = " << result << std::endl;
        //     std::cout << "输入要查询的w值 (输入q退出): ";
        // }
        
        // 示例积分计算
        double w1, w2;
        std::cout << "\n输入积分下限w1和上限w2 (用空格分隔): ";
        while (std::cin >> w1 >> w2) {
            try {
                double result = data.integrate(w1, w2);
                std::cout << "∫(从" << w1 << "到" << w2 << ") f(w) dw = " 
                          << result << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "错误: " << e.what() << std::endl;
            }
            std::cout << "\n输入积分下限w1和上限w2 (用空格分隔): ";
        }

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}