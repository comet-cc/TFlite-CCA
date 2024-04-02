// my_util_functions.h
#ifndef MY_UTIL_FUNCTIONS_H
#define MY_UTIL_FUNCTIONS_H
#include <string>
namespace tflite {
namespace label_image {

// Example function declaration
std::string checkSystemStateAndGetFilename(std::string filePath);
void updateSystemStateToProcessed(std::string filePath);
}  // namespace label_image
}  // namespace tflite

#endif  // MY_UTIL_FUNCTIONS_H
