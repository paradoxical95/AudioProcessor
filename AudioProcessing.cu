#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>

// WAV file header structure
struct WavHeader {
    char riff[4];         // "RIFF"
    uint32_t size;        // Size of the entire file - 8
    char wave[4];         // "WAVE"
    char fmt[4];          // "fmt "
    uint32_t fmtSize;     // Size of the fmt chunk
    uint16_t format;      // Format type
    uint16_t channels;    // Number of channels
    uint32_t sampleRate;  // Sampling rate
    uint32_t byteRate;    // (Sample Rate * Block Align)
    uint16_t blockAlign;  // (Channels * BitsPerSample/8)
    uint16_t bitsPerSample; // Bits per sample
    char data[4];         // "data"
    uint32_t dataSize;    // Size of the data section
};

// Function to handle CUDA errors
void checkCudaError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in " << context << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to read WAV file
void readWav(const char* filename, std::vector<float>& audioSamples, int& sampleRate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    // Debug: Check the raw header bytes
    std::cout << "Raw Header Bytes: ";
    for (size_t i = 0; i < sizeof(WavHeader); ++i) {
        std::cout << std::hex << (int)((char*)&header)[i] << " ";
    }
    std::cout << std::dec << std::endl;

    // Check if the WAV header is valid
    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Invalid WAV file format." << std::endl;
        return;
    }

    sampleRate = header.sampleRate;

    // Debug: Print sample rate and format info
    std::cout << "WAV Format: " << header.format << ", Channels: " << header.channels 
              << ", Sample Rate: " << sampleRate << ", Bits per Sample: " << header.bitsPerSample << std::endl;

    if (header.bitsPerSample != 24) {
        std::cerr << "Unsupported bits per sample: " << header.bitsPerSample << std::endl;
        return;
    }

    audioSamples.resize(header.dataSize / 3); // 24 bits = 3 bytes
    std::vector<unsigned char> buffer(header.dataSize);
    file.read(reinterpret_cast<char*>(buffer.data()), header.dataSize);

    // Convert 24-bit samples to float
    for (size_t i = 0; i < buffer.size() / 3; ++i) {
        // Read 3 bytes for each sample and combine them into a 24-bit value
        int32_t sample = (buffer[i * 3] << 0) | (buffer[i * 3 + 1] << 8) | (buffer[i * 3 + 2] << 16);
        audioSamples[i] = sample / 8388608.0f; // Convert to range [-1.0, 1.0]
    }

    // Debug: Check the number of samples read and some initial samples
    std::cout << "Number of Samples Read: " << audioSamples.size() << std::endl;
    std::cout << "First 10 Samples Read:" << std::endl;
    for (int i = 0; i < 10 && i < audioSamples.size(); ++i) {
        std::cout << "Sample " << i << ": " << audioSamples[i] << std::endl;
    }
}

// Function to write WAV file
void writeWav(const char* filename, const std::vector<float>& audioSamples, int sampleRate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    WavHeader header = {};
    std::copy("RIFF", "RIFF" + 4, header.riff);
    std::copy("WAVE", "WAVE" + 4, header.wave);
    std::copy("fmt ", "fmt " + 4, header.fmt);
    header.fmtSize = 16;
    header.format = 1; // PCM
    header.channels = 2; // Stereo (2 channels)
    header.sampleRate = sampleRate;
    header.bitsPerSample = 24; // Set to 24 bits
    header.byteRate = sampleRate * header.channels * header.bitsPerSample / 8;
    header.blockAlign = header.channels * header.bitsPerSample / 8;

    // Calculate data size
    header.dataSize = audioSamples.size() * header.channels * (header.bitsPerSample / 8);
    header.size = 36 + header.dataSize;
    std::copy("data", "data" + 4, header.data);

    // Write header
    file.write(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    if (!file) {
        std::cerr << "Error writing WAV header to file " << filename << std::endl;
        return;
    }

    // Write audio samples
    for (size_t i = 0; i < audioSamples.size(); ++i) {
        // Convert float to 24-bit integer (3 bytes)
        int32_t sample = static_cast<int32_t>(audioSamples[i] * 8388608.0f); // Convert float to 24-bit PCM

        // Write the sample as 3 bytes
        file.put((sample >> 0) & 0xFF);
        file.put((sample >> 8) & 0xFF);
        file.put((sample >> 16) & 0xFF);

        // Debug: Print float and converted sample value
        std::cout << "Float Sample " << i << ": " << audioSamples[i] << " | Converted Sample: " << sample << std::endl;

        // Check if write was successful
        if (!file) {
            std::cerr << "Error writing sample " << i << " to file " << filename << std::endl;
            break; // Exit loop on write failure
        }
    }

    // Debug: Check the file size after writing
    file.seekp(0, std::ios::end);
    std::streampos size = file.tellp();
    std::cout << "Output file size: " << size << " bytes" << std::endl;

    // Close the file
    file.close();
}


// CUDA kernel for applying reverb
__global__ void applyReverbKernel(float* audio, float* impulse, float* output, int audioSize, int impulseSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < audioSize) {
        float sample = 0.0f;
        for (int j = 0; j < impulseSize; ++j) {
            if (idx - j >= 0) {
                sample += audio[idx - j] * impulse[j];
            }
        }
        output[idx] = sample;
    }
}

// CUDA kernel for applying equalization
__global__ void applyEQKernel(float* audio, float* filter, float* output, int audioSize, int filterSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < audioSize) {
        float sample = 0.0f;
        for (int j = 0; j < filterSize; ++j) {
            if (idx - j >= 0) {
                sample += audio[idx - j] * filter[j];
            }
        }
        output[idx] = sample;
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.wav>" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = "output.wav";
    std::vector<float> audioSamples;
    int sampleRate;

    // Read audio file
    readWav(inputFile, audioSamples, sampleRate);

    // Allocate memory on the GPU
    float *d_audio, *d_impulse, *d_output, *d_eqFilter;
    int impulseSize = 44100; // Example impulse response length (1 second at 44.1 kHz)
    std::vector<float> impulse(impulseSize, 0.01f); // Simple impulse response
    std::vector<float> eqFilter = { 0.5f, 0.5f }; // Example simple EQ filter (2-tap)

    checkCudaError(cudaMalloc(&d_audio, audioSamples.size() * sizeof(float)), "cudaMalloc d_audio");
    checkCudaError(cudaMalloc(&d_impulse, impulse.size() * sizeof(float)), "cudaMalloc d_impulse");
    checkCudaError(cudaMalloc(&d_output, audioSamples.size() * sizeof(float)), "cudaMalloc d_output");
    checkCudaError(cudaMalloc(&d_eqFilter, eqFilter.size() * sizeof(float)), "cudaMalloc d_eqFilter");

    // Copy data to GPU
    checkCudaError(cudaMemcpy(d_audio, audioSamples.data(), audioSamples.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_audio");
    checkCudaError(cudaMemcpy(d_impulse, impulse.data(), impulse.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_impulse");
    checkCudaError(cudaMemcpy(d_eqFilter, eqFilter.data(), eqFilter.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_eqFilter");

    // Launch kernels
    int blockSize = 256;
    int numBlocks = (audioSamples.size() + blockSize - 1) / blockSize;

    applyReverbKernel<<<numBlocks, blockSize>>>(d_audio, d_impulse, d_output, audioSamples.size(), impulse.size());
    checkCudaError(cudaGetLastError(), "launch applyReverbKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after reverb");

    applyEQKernel<<<numBlocks, blockSize>>>(d_output, d_eqFilter, d_audio, audioSamples.size(), eqFilter.size());
    checkCudaError(cudaGetLastError(), "launch applyEQKernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after EQ");

    // Copy output back to host
    checkCudaError(cudaMemcpy(audioSamples.data(), d_output, audioSamples.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    // Write output WAV file
    writeWav(outputFile, audioSamples, sampleRate);

    // Free GPU memory
    cudaFree(d_audio);
    cudaFree(d_impulse);
    cudaFree(d_output);
    cudaFree(d_eqFilter);

    return 0;
}
