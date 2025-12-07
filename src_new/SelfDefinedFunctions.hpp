// SelfDefinedFunctions.hpp
#ifndef SELFDEFINEDFUNCTIONS_HPP
#define SELFDEFINEDFUNCTIONS_HPP
#include <string>
#include <vector>

std::vector<std::string> getFilesInFolder(const std::string& folderPath) {}

void writeVectorToFile(const std::vector<int>& vec, const std::string& filename) {}

std::vector<int> readVectorFromFile(const std::string& filename) {}

double getmax_2(double a, double b) {}

double getmax_3(double a, double b, double c) {}

int sort_by_row(const void* a, const void* b) {}

#endif        // SELFDEFINEDFUNCTIONS_HPP