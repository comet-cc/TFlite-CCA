#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include "tensorflow/lite/examples/label_image/VM_signalling.h"

std::string checkSystemStateAndGetFilename(std::string filePath) {
    while (true) {
        std::ifstream file(filePath);
        std::string line;
        std::string state;
        std::string fileName;

        while (getline(file, line)) {
            if (line.find("systemState:") != std::string::npos) {
                state = line.substr(line.find(":") + 2);
            } else if (line.find("fileName:") != std::string::npos) {
                fileName = line.substr(line.find(":") + 2);
            }
	 std::cout << "Waiting for signalling file..." << std::endl; 
        }

        if (state == "query") {
            std::cout << "System state is 'query'. Filename: " << fileName << std::endl;
            return fileName;
        } else {
            std::cout << "Waiting for system state to become 'query'..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5)); // Check every 5 seconds
        }
    }
}

void updateSystemStateToProcessed(std::string filePath) {
    std::ofstream file(filePath, std::ios::out | std::ios::trunc); // Overwrite the file
    if (file.is_open()) {
        file << "systemState: processed\n";
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

