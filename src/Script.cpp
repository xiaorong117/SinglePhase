#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main() {
    std::string filename = "TEST64nm.txt"; // 文件名
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return 1;
    }

    std::vector<std::string> thirdColumnData; // 存储第三列数据
    std::string line;
    
    // 逐行读取文件
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string col1, col2, col3, col4, col5;
        
        // 尝试提取四列数据
        if (iss >> col1 >> col2 >> col3 >> col4 >> col5) {
            thirdColumnData.push_back(col4);
        }
    }
    
    file.close();

    // 格式化为 {value1,value2,value3...} 输出
    std::cout << "{";
    for (size_t i = 0; i < 50; ++i) {
        std::cout << thirdColumnData[i];
        if (i != thirdColumnData.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "}" << std::endl;

    return 0;
}