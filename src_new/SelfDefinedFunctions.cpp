#include "SelfDefinedFunctions.hpp"
#include <dirent.h>           // ✅ 目录操作函数 (opendir, readdir, closedir)
#include <sys/types.h>        // 建议包含以增强可移植性
#include <filesystem>         // 您代码中已经包含，但未在函数中使用
#include <fstream>
#include <iostream>
#include "Globals.hpp"
// #include <cstring>   // 可选，如果遇到 C 字符串相关的编译警告

using namespace std;

std::vector<std::string> getFilesInFolder(const std::string& folderPath) {
  std::vector<std::string> fileList;

  DIR* dir;
  struct dirent* entry;

  // 打开文件夹
  dir = opendir(folderPath.c_str());
  if (dir == nullptr) {
    return fileList;
  }

  // 读取文件夹中的文件
  while ((entry = readdir(dir)) != nullptr) {
    // 忽略当前目录和上级目录
    if (std::string(entry->d_name) == "." || std::string(entry->d_name) == "..") {
      continue;
    }

    // 将文件名添加到列表中
    fileList.push_back(entry->d_name);
  }

  // 关闭文件夹
  closedir(dir);

  return fileList;
}

void writeVectorToFile(const std::vector<int>& vec, const std::string& filename) {
  std::ofstream outFile(filename);
  if (!outFile) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return;
  }

  for (const auto& num : vec) {
    outFile << num << " ";        // 用空格分隔数字
  }

  outFile.close();
  std::cout << "vector已写入文件: " << filename << std::endl;
}

std::vector<int> readVectorFromFile(const std::string& filename) {
  std::vector<int> vec;
  std::ifstream inFile(filename);
  if (!inFile) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return vec;        // 返回空vector
  }

  int num;
  while (inFile >> num) {
    vec.push_back(num);
  }

  inFile.close();
  std::cout << "vector已从文件读取: " << filename << std::endl;
  return vec;
}

double getmax_2(double a, double b) {
  return a > b ? a : b;
}

double getmax_3(double a, double b, double c) {
  double temp = getmax_2(a, b);
  temp = getmax_2(temp, c);
  return temp;
}

int sort_by_row(const void* a, const void* b) {
  if (((Acoo*)a)->row != ((Acoo*)b)->row) {
    return ((Acoo*)a)->row > ((Acoo*)b)->row;
  } else {
    return ((Acoo*)a)->col > ((Acoo*)b)->col;
  }
}        // namespace Solver_property