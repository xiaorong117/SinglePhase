#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip> // 用于格式化输出

// 进度条显示函数
void showProgress(float progress, int barWidth = 50) {
    std::cout << "[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%\r";
    std::cout.flush();
}

// 计算文件行数
size_t countLines(std::ifstream& file) {
    size_t lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        lines++;
    }
    file.clear();
    file.seekg(0, std::ios::beg);
    return lines;
}

int main() {
    std::ifstream file1("Tb_debug.txt");
    std::ifstream file2("Tb_debug0.txt");
    std::ofstream diff_out("diff_results.txt");

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return 1;
    }

    // 计算文件行数用于进度显示
    size_t totalLines = countLines(file1);
    size_t totalLines2 = countLines(file2);
    size_t minLines = std::min(totalLines, totalLines2);
    std::cout << "Total lines: " << minLines << "\nComparing files..." << std::endl;

    std::string line1, line2;
    size_t lineNum = 0;
    bool mismatchFound = false;

    while (std::getline(file1, line1) && std::getline(file2, line2)) {
        ++lineNum;
        
        // 更新进度条
        if (lineNum % 1000 == 0 || lineNum == minLines) {
            showProgress(static_cast<float>(lineNum) / minLines);
        }
        
        // 解析行数据
        std::istringstream iss1(line1);
        std::istringstream iss2(line2);
        int index1, id1_1, id2_1;
        int index2, id1_2, id2_2;
        double cond1, cond2;
        
        if (!(iss1 >> index1 >> id1_1 >> id2_1 >> cond1)) {
            std::cerr << "\nFormat error at line " << lineNum << " in file1" << std::endl;
            return 2;
        }
        
        if (!(iss2 >> index2 >> id1_2 >> id2_2 >> cond2)) {
            std::cerr << "\nFormat error at line " << lineNum << " in file2" << std::endl;
            return 3;
        }

        // 计算差值并写入输出
        diff_out << (index1 - index2) << "\t"
                << (id1_1 - id1_2) << "\t"
                << (id2_1 - id2_2) << "\t"
                << (cond1 - cond2) << std::endl;
                
        // 检查是否有差异
        if (index1 != index2 || id1_1 != id1_2 || id2_1 != id2_2 || 
            std::abs(cond1 - cond2) > 1e-9) {
            mismatchFound = true;
        }
    }

    // 完成进度条
    showProgress(1.0f);
    std::cout << "\n"; // 换行

    // 检查文件长度
    bool moreIn1 = false, moreIn2 = false;
    if (std::getline(file1, line1)) moreIn1 = true;
    if (std::getline(file2, line2)) moreIn2 = true;
    
    if (moreIn1 || moreIn2) {
        std::cerr << "WARNING: Files have different line counts!" << std::endl;
        std::cerr << "File1 has " << totalLines << " lines" << std::endl;
        std::cerr << "File2 has " << totalLines2 << " lines" << std::endl;
        std::cerr << "Compared only first " << minLines << " lines" << std::endl;
    }

    std::cout << "Comparison complete. Results saved to diff_results.txt" << std::endl;
    if (!mismatchFound) {
        std::cout << "✔️ All values match exactly." << std::endl;
    } else {
        std::cout << "⚠️ Some values differ. Check diff_results.txt for details." << std::endl;
    }

    file1.close();
    file2.close();
    diff_out.close();

    return 0;
}