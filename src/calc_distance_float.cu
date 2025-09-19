
#include <cuda_runtime.h>
#include <cuda.h>    
#include "common.h"  //-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/    ~.h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <vector>
#include <string>
#include <array>
#include <math.h>
#include <tuple>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdlib.h>
#include "numpy.hpp"

#include <filesystem>  
#include <random>
#include <omp.h>  //OpenMP


//nvc++ -ta=tesla calc_distance_float.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o cal_d
//./cal_d /home/gpu_data/data7/ 1st cfos_CT0_01 dist_v psf mask 0 11 10 81

//CUDA error check
#define CHECK(call)                   \
{                                      \
  const cudaError_t error = call;      \
  if(error != cudaSuccess)             \
  {                                    \
      printf("Error: %s:%d, ", __FILE__, __LINE__);               \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));    \
      exit(1);        \
  }                \
}             \



double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


__global__ void calc_dist_gpu(int N, float* data, double* psf, int* mask, float* distances, float* min_dist, int size, int psf_num, int dist_num){

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;   ,dist = N*10
    // unsigned int idx =  ix *y_b_num * z_b_num +  iy * z_b_num + iz;

    if (ix < N) {
        if (iy < psf_num){
            for (int i = 0; i < size ; i += 1) {
                for (int j = 0; j < size ; j += 1) {
                    for (int k = 0; k < size ; k += 1) {

                        if (mask[i*size*size+ j*size+k] > 0){
                            if ((data[ix*size*size*size + i*size*size+ j*size+k] > 0)&(psf[iy*size*size*size + i*size*size+ j*size+k]> 0)){
                                
                                float m_ratio = data[ix*size*size*size + i*size*size+ j*size+k] / psf[iy*size*size*size + i*size*size+ j*size+k];
                                float s_ratio = (data[ix*size*size*size + i*size*size+ j*size+k] + psf[iy*size*size*size + i*size*size+ j*size+k]);

                                // All metrics that require positive non-zero values
                                distances[ix*psf_num*dist_num+iy*dist_num+0] += std::pow((data[ix*size*size*size + i*size*size+ j*size+k] * psf[iy*size*size*size + i*size*size+ j*size+k]), 0.5);  // Hellinger Coefficient
                                distances[ix*psf_num*dist_num+iy*dist_num+1] += std::pow(data[ix*size*size*size + i*size*size+ j*size+k], data[ix*size*size*size + i*size*size+ j*size+k]) * std::pow(psf[iy*size*size*size + i*size*size+ j*size+k], (1.0 - data[ix*size*size*size + i*size*size+ j*size+k]));  // Chernoff Coefficient
                                distances[ix*psf_num*dist_num+iy*dist_num+3] += data[ix*size*size*size + i*size*size+ j*size+k] * std::log2(m_ratio);  // Directed Divergence
                                distances[ix*psf_num*dist_num+iy*dist_num+4] += (data[ix*size*size*size + i*size*size+ j*size+k] - psf[iy*size*size*size + i*size*size+ j*size+k]) * std::log2(m_ratio) ; // J-Divergence
                                distances[ix*psf_num*dist_num+iy*dist_num+7] += data[ix*size*size*size + i*size*size+ j*size+k] * std::log10(m_ratio);  // KL Divergence
                                // distances[ix*psf_num*dist_num+iy*dist_num+7] += std::pow((data[ix*size*size*size + i*size*size+ j*size+k]-psf[iy*size*size*size + i*size*size+ j*size+k]),3)/std::pow(mask[i*size*size+ j*size+k],3) ;  // KM Divergence

                                if (s_ratio > 0){
                                    distances[ix*psf_num*dist_num+iy*dist_num+9] += data[ix*size*size*size + i*size*size+ j*size+k] * std::log10(2.0 * data[ix*size*size*size + i*size*size+ j*size+k] / s_ratio) +
                                                    psf[iy*size*size*size + i*size*size+ j*size+k] * std::log10(2.0 * psf[iy*size*size*size + i*size*size+ j*size+k] / s_ratio);  //# JS Divergence
                                }
                            }
                            distances[ix*psf_num*dist_num+iy*dist_num+2] += std::pow((std::pow(data[ix*size*size*size + i*size*size+ j*size+k], 0.5) - std::pow(psf[iy*size*size*size + i*size*size+ j*size+k], 0.5)), 2.0);  // Jeffreys Distance
                            distances[ix*psf_num*dist_num+iy*dist_num+5] += abs(data[ix*size*size*size + i*size*size+ j*size+k] - psf[iy*size*size*size + i*size*size+ j*size+k]);  // L1 norm
                            distances[ix*psf_num*dist_num+iy*dist_num+6] += std::pow((data[ix*size*size*size + i*size*size+ j*size+k] - psf[iy*size*size*size + i*size*size+ j*size+k]), 2);  // L2 norm squared
                            
                            if (data[ix*size*size*size + i*size*size+ j*size+k] != 0){
                            distances[ix*psf_num*dist_num+iy*dist_num+8] += std::pow((data[ix*size*size*size + i*size*size+ j*size+k] - psf[iy*size*size*size + i*size*size+ j*size+k]), 2) / data[ix*size*size*size + i*size*size+ j*size+k];  // PE
                            }else{
                                distances[ix*psf_num*dist_num+iy*dist_num+8] +=0;
                            }
                        }
                    }
                }
            }
            distances[ix*psf_num*dist_num+iy*dist_num+6] = std::pow(distances[ix*psf_num*dist_num+iy*dist_num+6], 0.5);  // Convert L2 norm squared to L2 norm
        
            
            }//iy
             __syncthreads();

             for (int n = 0; n < dist_num ; n += 1) {
                float min = min_dist[ix*dist_num+n];
                for (int m = 0; m < psf_num ; m += 1) {
                    min = std::fmin(min, distances[ix*psf_num*dist_num+m*dist_num+n]);
                }
                min_dist[ix*dist_num+n] = min;
             }

        }
    }


std::string getLeftPaddingString(std::string const &str, int n, char paddedChar = ' ')
{
    std::ostringstream ss;
    ss << std::right << std::setfill(paddedChar) << std::setw(n) << str;
    return ss.str();
}

bool fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}



int main(int argc, char * argv[]){
int g_c=1;
//importing bin file


std::string savedir = argv[g_c];
g_c +=1;

std::string exp = argv[g_c];
g_c +=1;

std::string dir = argv[g_c];
g_c +=1;

std::string outf = argv[g_c];
g_c +=1;

std::string psf_file = argv[g_c];
g_c +=1;

std::string mask_file = argv[g_c];
g_c +=1;

std::string points_num = argv[g_c]; //"cell_table_combine.npy";
g_c +=1;

int size = atoi(argv[g_c]);  //11
g_c +=1;

int dist_num =10;

printf("outf %s\n", outf.c_str());


int blockdim_x =atoi(argv[g_c]); //cp
g_c +=1;
int blockdim_y =atoi(argv[g_c]); //cp
g_c +=1;
// int blockdim_y =atoi(argv[g_c]);  //N
// g_c +=1;
// int blockdim_z =atoi(argv[g_c]);  //N
// g_c +=1;

printf("blockdim_x  %d\n", blockdim_x);


//start timer
double iStart = cpuSecond();
//GPU device setup
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

    
    std::string outfol =  savedir  +exp + "/"+dir+"/";
    std::filesystem::create_directories(outfol); 

    std::string save_f= outfol + outf + points_num+".bin";

    // if (fileExists(save_f)) {
    //     std::cout << save_f << " exists" << std::endl;
    // }else{
        
        std::vector<int> s0;
        std::vector<float> data;  
        std::string roi_f = savedir  +exp + "/"+dir+"/roi_norm"+points_num+".npy";

        std::cout << roi_f << std::endl;

        aoba::LoadArrayFromNumpy(roi_f, s0, data);
        std::cout << "size:" << s0[0] << " " << s0[1] << std::endl;

        printf("i: %d ,  roi_vs:%f\n", 0, data[0]);

        int N = s0[0];
        int col_num = s0[1];  //1   z, y, x
        printf("total points number  %d\n", N);
        printf("col_num  %d\n",col_num );
        
        // float *data2=(float*)malloc(sizeof(float)*N*col_num);
      
            // for (int i = 0; i < N ; i += 1) {
            //     for (int j = 0; j < col_num ; j += 1) {
                
            //         data2[i*col_num + j] = data[i*col_num + j];
                
            //     }
            //     //  if (i<3){
            //     // printf("i: %d ,  vxx[i]:%d\n", i, vxx[i]);
            //     // printf("i: %d ,  vxy[i]:%d\n", i, vxy[i]);
            //     // printf("i: %d ,  xvz[i]:%d\n", i, vxz[i]);
            //     //  }
            //    }
               

            std::vector<int> s2;
            std::vector<int> mask;  
            std::string mask_f=  "/home/gpu_data/data7/cfos_roi/" +mask_file+".npy";  // cell coordinates after ANTs points
            std::cout << mask_f << std::endl;
            aoba::LoadArrayFromNumpy(mask_f, s2, mask);
            std::cout << "size:" << s2[0]  << std::endl;
            printf("i: %d ,  mask[0]:%d\n", 0, mask[0]);
            int mask_num = s2[0];
            // int psf_num = s[0];
            printf("mask_size,  %d\n", mask_num);
            // printf("cell_count,  %d\n", cell_count);



            std::vector<int> s;
            std::vector<double> psf;  
            std::string psf_f= "/home/gpu_data/data7/cfos_roi/" +psf_file+".npy";  // cell coordinates after ANTs points
            std::cout << psf_f << std::endl;
            aoba::LoadArrayFromNumpy(psf_f, s, psf);
            std::cout << "size:" << s[0] << " " << s[1] << std::endl;
            printf("i: %d ,  psf[0]:%f\n", 0, psf[0]);
            int psf_num = s[0];
            // int psf_num = s[0];
            printf("psf_num,  %d\n", psf_num);
    

            // for (int i = 0; i < cell_count ; i += 1) {
            //     for (int j = 0; j < col_num ; j += 1) {
            //     if (j==0){
            //     x[i] = static_cast<int>(cell_cords[j*cell_count+i]);
            //     }else if(j==1){
            //     y[i] = static_cast<int>(cell_cords[j*cell_count+i]);
            //     }else if(j==2){
            //     z[i] = static_cast<int>(cell_cords[j*cell_count+i]);
            //     }

            //     // if (i<3){
            //     // printf("i: %d ,  x[i]:%d\n", i, x[i]);
            //     // printf("i: %d ,  y[i]:%d\n", i, y[i]);
            //     // printf("i: %d ,  z[i]:%d\n", i, z[i]);
            //     // }
            //   }
            // }


            // int *distances=(float*)malloc(sizeof(float)*N*psf_num*dist_num);
            float *min_dist=(float*)malloc(sizeof(float)*N*dist_num);
            // double *vx_count_ratio=(double*)malloc(sizeof(double)*vx_num);

            for (int i = 0; i < N ; i += 1) {
                for (int j = 0; j < dist_num ; j += 1) {
                min_dist[i*dist_num+j] = std::numeric_limits<float>::infinity();
                }
            }

            // printf("a\n");
            // 
            // printf("aa\n");
            //device memory
            float *d_data;
            int *d_mask;
            double *d_psf ;
            float *d_distances;
            float *d_min_dist;
            CHECK(cudaMalloc((void **)&d_data, sizeof(float)*N*size*size*size));
            CHECK(cudaMalloc((void **)&d_psf, sizeof(double)*psf_num * size*size*size));
            CHECK(cudaMalloc((void **)&d_mask, sizeof(int)*size*size*size));
            CHECK(cudaMalloc((void **)&d_distances, sizeof(float)*N*psf_num*dist_num));
            CHECK(cudaMalloc((void **)&d_min_dist, sizeof(float)*N*dist_num));
       


            // printf("aaa\n");

            //memcopy host -> device
            CHECK(cudaMemcpy(d_data, data.data(), sizeof(float)*N*size*size*size, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_psf, psf.data(), sizeof(double)*psf_num * size*size*size, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_mask, mask.data(), sizeof(int)*size*size*size, cudaMemcpyHostToDevice));    
            
            CHECK(cudaMemcpy(d_min_dist, min_dist, sizeof(float)*N*dist_num, cudaMemcpyHostToDevice));
  


                // if (x_b_num<1024){
                //     blockdim_x=x_b_num;
                // }
                // if (y_b_num<1024){
                //     blockdim_y=y_b_num;
                // }
                // if (z_b_num<1024){
                //     blockdim_z=z_b_num;
                // }

                int dimx =N/blockdim_x;// (NE+NI)/blockdim_x;
                int dimy = psf_num/blockdim_y;
                // int dimz = z_b_num/blockdim_z;
                dim3 block(blockdim_x, blockdim_y);
                dim3 grid(dimx, dimy);

                

                calc_dist_gpu<<<grid, block>>>(N, d_data, d_psf, d_mask, d_distances, d_min_dist, size, psf_num, dist_num);

                CHECK(cudaDeviceSynchronize());
                CHECK(cudaGetLastError());
                printf("end vx_count\n");
            
            

                        // copy kernel result back to host side
                
                CHECK(cudaMemcpy(min_dist, d_min_dist, sizeof(float)*N*dist_num,  cudaMemcpyDeviceToHost));

                for (int i = 0; i < N ; i += 1) {
                    for (int j = 0; j < dist_num ; j += 1) {
                    if ((i<3)){
                    printf("i: %d, j: %d,  min_dist:%f\n", i, j, min_dist[i*dist_num + j]);
                    }
                 }
                }
                
            //savefile
                
            //savefiles
            std::ofstream ofs;
                ofs.open(save_f, std::ios::out|std::ios::binary|std::ios::trunc);
                if (!ofs) {
                std::cout << "Can't open a file"<<save_f<<std::endl;
                }
                    
                for (int i = 0; i < N ; i += 1) {
                    for (int j = 0; j < dist_num ; j += 1) {
                    ofs.write(( char * ) &min_dist[i*dist_num + j], sizeof(float) );
                    // printf("count %d, cv %f\n", i, cvs[i]);
                    }//for
                }
                ofs.close();

            CHECK(cudaFree(d_data));
            CHECK(cudaFree(d_mask));
            CHECK(cudaFree(d_psf));
            CHECK(cudaFree(d_distances));
            CHECK(cudaFree(d_min_dist));
        

            // free(x);
            // free(y);
            // free(z);
            free(min_dist);
            // free(vx_count_ratio);
// free(spc);

        // }
    
   

   //end timer
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
}






// 

