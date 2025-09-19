
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
#include <stdlib.h>
#include "numpy.hpp"

#include <filesystem>  
#include <random>
#include <omp.h>  //OpenMP

//for voxel based phase in whole brain
//nvc++ -mp -ta=tesla cal_vx_count_r_whole.cu -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/examples/OpenACC/SDK/include/ -I/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/include -L/tmp/nvhpc_2023_233_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/23.3/math_libs/12.0/targets/x86_64-linux/lib -lcudart -std=c++20 -o vx_c_w
//./vx_c_w 2 2 2 144 197 175  SCH 50 3 1 0.5 /home/gpu_data/data8/cfos_app ANTsR50 2 2 2
//./vx_cr 153 163 82 114 165 157 SCH 50 3 1 0.5 /home/gpu_data/data8/cfos_app ANTsR50 2 8 8 8

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


__global__ void vx_gpu(int* vxx, int* vxy,  int* vxz, int vx_num, int *x, int *y,  int *z, int *vx_count, int cell_count,  int vx, int vb_r){

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned int iz = threadIdx.z + blockIdx.z * blockDim.z;
    // unsigned int idx =  ix *y_b_num * z_b_num +  iy * z_b_num + iz;

    if (ix < vx_num) {

                int count=0;
                // printf("count %d, N %d", ix, iy);
                for (int it=0; it<cell_count; it++){
                    // if ((ix<1)&(iy<1)&(iy<1)&(it<3)){
                    // printf("it %d, x %d, y %d, z %d \n", it, x[it], y[it], z[it]);
                    // printf("xrange low:%d\n", xmin*vx+ix*mo*vx-int(vb_r*vx/2));
                    // printf("xrange high:%d\n", xmin*vx+ix*mo*vx+int(vb_r*vx/2));
                    // }
                    if((x[it]>=vxx[ix]*vx-int(vb_r*vx/2))&(x[it]<vxx[ix]*vx+int(vb_r*vx/2))&(y[it]>=vxy[ix]*vx-int(vb_r*vx/2))&(y[it]<vxy[ix]*vx+int(vb_r*vx/2))&(z[it]>=vxz[ix]*vx-int(vb_r*vx/2))&(z[it]<vxz[ix]*vx+int(vb_r*vx/2))){
                    count+=1;
                      }//if(x~)
                    } //for  cell_count
                vx_count[ix] = count;
                // if ((ix<3)&(iy<3)&(iy<3)){
                // printf("count %d\n", count);
                // printf("idx %d\n", idx);
                // }
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

// int cell_count = atoi(argv[g_c]); 
// g_c +=1;

// std::string region = argv[g_c];
// g_c +=1;
int vx = atoi(argv[g_c]); 
g_c +=1;
int vb_r = atoi(argv[g_c]); 
g_c +=1;
int mo = atoi(argv[g_c]); 
g_c +=1;

std::string vx_cords_f = argv[g_c];
g_c +=1;

// std::cout << r_str << std::endl;
// printf("r %s\n", r_str.c_str());

std::string savedir = argv[g_c];
g_c +=1;

std::string root_fol = argv[g_c];
g_c +=1;

std::string ants_dir_name = argv[g_c];
g_c +=1;

std::string combine_points_f = argv[g_c];//"cell_table_combine.npy";
g_c+=1;

int cpu_num = atoi(argv[g_c]);
g_c +=1;

printf("vb_r,  %d\n", vb_r);
printf("ants file %s\n", ants_dir_name.c_str());


int blockdim_x =atoi(argv[g_c]); //cp
g_c +=1;


printf("blockdim_x  %d\n", blockdim_x);

//read cell coordinates file
int CT_num = 48;
std::vector<int> CT_li;
for (int i = 0; i < CT_num ; i += 4) {
    CT_li.push_back(i);
}

std::vector<int> sample_ids;
for (int i = 1; i < 7 ; i += 1) {
    sample_ids.push_back(i);
}

// std::string sample = "CT0_01";
std::vector<std::string> exps{"1st", "2nd"} ;

//start timer
double iStart = cpuSecond();
//GPU device setup
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

for (int l = 0; l < exps.size() ; l += 1) {
   
    std::string exp = exps[l];
    printf(" %s\n", exp.c_str());

    #pragma omp parallel for num_threads(cpu_num)
    for (int m = 0; m < CT_li.size() ; m += 1) {
        
        for (int n = 0; n < sample_ids.size() ; n += 1) {
            std::string pd = getLeftPaddingString(std::to_string(sample_ids[n]), 2, '0')  ;

            std::string sample = "CT"+std::to_string(CT_li[m])+ "_" + pd;
            printf(" %s\n", sample.c_str());

            std::string root_vx_f = savedir + "/" +exp + "/"+root_fol+"/"+std::to_string(vx)+"um/whole/vb"+ std::to_string(vb_r)+"_mo"+std::to_string(mo)+"/";
            std::filesystem::create_directories(root_vx_f); 

            std::string vx_f= root_vx_f + sample + "_vb_CT_"+exp+".bin";

            if (fileExists(vx_f)) {
                std::cout << vx_f << " exists" << std::endl;
            }else{

            //voxel coordinates from atlas
            std::vector<int> s0;
            std::vector<int> vx_cords;  
            std::string vx_npy = vx_cords_f;  // vx coordinates of brains

            std::cout << vx_npy << std::endl;

            aoba::LoadArrayFromNumpy(vx_npy, s0, vx_cords);
            std::cout << "size:" << s0[0] << " " << s0[1] << std::endl;

            printf("i: %d ,  vx_cords[0]:%d\n", 0, vx_cords[0]);

            int vx_num = s0[1];
            int col_num_vx = s0[0];
            printf("total voxel  %d\n", vx_num);
            std::vector<int> vxx(sizeof(int)*vx_num);
            std::vector<int> vxy(sizeof(int)*vx_num);
            std::vector<int> vxz(sizeof(int)*vx_num);


                for (int i = 0; i < vx_num ; i += 1) {
                    for (int j = 0; j < col_num_vx ; j += 1) {
                    if (j==0){
                    vxx[i] = vx_cords[j*vx_num+i];
                    }else if(j==1){
                    vxy[i] = vx_cords[j*vx_num+i];
                    }else if(j==2){
                    vxz[i] = vx_cords[j*vx_num+i];
                    }
                    if (i<3){
                    printf("i: %d ,  vxx[i]:%d\n", i, vxx[i]);
                    printf("i: %d ,  vxy[i]:%d\n", i, vxy[i]);
                    printf("i: %d ,  xvz[i]:%d\n", i, vxz[i]);
                    }
                }
                }  




            std::vector<int> s;
            std::vector<double> cell_cords;  
            std::string cell_npy = savedir +"/"+ exp+"/" +  sample +"/SYTOX-G/"+ants_dir_name +"/"+combine_points_f;
            aoba::LoadArrayFromNumpy(cell_npy, s, cell_cords);
            std::cout << "size:" << s[0] << " " << s[1] << std::endl;

            printf("i: %d ,  cell_cords[0]:%f\n", 0, cell_cords[0]);

            int cell_count = s[1];
            int col_num = s[0];

            printf("col_num,  %d\n", col_num);
            printf("cell_count,  %d\n", cell_count);

            // int *x=(int*)malloc(sizeof(int)*cell_count);
            // int *y=(int*)malloc(sizeof(int)*cell_count);
            // int *z=(int*)malloc(sizeof(int)*cell_count);

            std::vector<int> x(sizeof(int)*cell_count);
            std::vector<int> y(sizeof(int)*cell_count);
            std::vector<int> z(sizeof(int)*cell_count);


            for (int i = 0; i < cell_count ; i += 1) {
                for (int j = 0; j < col_num ; j += 1) {
                if (j==0){
                x[i] = static_cast<int>(cell_cords[j*cell_count+i]);
                }else if(j==1){
                y[i] = static_cast<int>(cell_cords[j*cell_count+i]);
                }else if(j==2){
                z[i] = static_cast<int>(cell_cords[j*cell_count+i]);
                }

                // if (i<3){
                // printf("i: %d ,  x[i]:%d\n", i, x[i]);
                // printf("i: %d ,  y[i]:%d\n", i, y[i]);
                // printf("i: %d ,  z[i]:%d\n", i, z[i]);
                // }
            }
            }

            if ((l==0)&(m==0)&(n==0)){
                for (int i = 0; i < 2 ; i += 1) {
                    printf("i: %d ,  x[i]:%d\n", i, x[i]);
                    printf("i: %d ,  y[i]:%d\n", i, y[i]);
                    printf("i: %d ,  z[i]:%d\n", i, z[i]);
                }
            }



            int *vx_count=(int*)malloc(sizeof(int)*vx_num);
            // double *vx_count_ratio=(double*)malloc(sizeof(double)*vx_num);

            // printf("a\n");
            // 


            // printf("aa\n");
            //device memory
            int *d_vxx,  *d_vxy, *d_vxz;
            CHECK(cudaMalloc((void **)&d_vxx, sizeof(int)*vx_num));
            CHECK(cudaMalloc((void **)&d_vxy, sizeof(int)*vx_num));
            CHECK(cudaMalloc((void **)&d_vxz, sizeof(int)*vx_num));

            int *d_x,  *d_y, *d_z, *d_vx_count;
            CHECK(cudaMalloc((void **)&d_x, sizeof(int)*cell_count));
            CHECK(cudaMalloc((void **)&d_y, sizeof(int)*cell_count));
            CHECK(cudaMalloc((void **)&d_z, sizeof(int)*cell_count));
            CHECK(cudaMalloc((void **)&d_vx_count, sizeof(int)*vx_num));


            // printf("aaa\n");

            //memcopy host -> device
            CHECK(cudaMemcpy(d_vxx, vxx.data(), sizeof(int)*vx_num, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_vxy, vxy.data(), sizeof(int)*vx_num, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_vxz, vxz.data(), sizeof(int)*vx_num, cudaMemcpyHostToDevice));    
            

            CHECK(cudaMemcpy(d_x, x.data(), sizeof(int)*cell_count, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_y, y.data(), sizeof(int)*cell_count, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_z, z.data(), sizeof(int)*cell_count, cudaMemcpyHostToDevice));    
            CHECK(cudaMemcpy(d_vx_count, vx_count, sizeof(int)*vx_num, cudaMemcpyHostToDevice));   

               

                int dimx =vx_num/blockdim_x;// (NE+NI)/blockdim_x;
                // int dimy = y_b_num/blockdim_y;
                // int dimz = z_b_num/blockdim_z;
                dim3 block(blockdim_x);
                dim3 grid(dimx);

                

                vx_gpu<<<grid, block>>>(d_vxx, d_vxy,  d_vxz, vx_num, d_x, d_y,  d_z, d_vx_count,  cell_count,  vx, vb_r);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaGetLastError());
                printf("end vx_count\n");
            
            

                        // copy kernel result back to host side
                
                CHECK(cudaMemcpy(vx_count, d_vx_count, sizeof(int)*vx_num,  cudaMemcpyDeviceToHost));


                 for (int i=0; i<vx_num; i++){
                    // vx_count_ratio[i] = static_cast<int>(vx_count[i]/cell_count);
                    if ((i<3)&(l==0)&(m==0)&(n==0)){
                    printf("i: %d ,  vx_count[i]:%d\n", i, vx_count[i]);
                    }
                 }
                
            //savefile
                
            //savefiles
            
            
            std::ofstream ofs;
                ofs.open(vx_f, std::ios::out|std::ios::binary|std::ios::trunc);
                if (!ofs) {
                std::cout << "Can't open a file"<<vx_f<<std::endl;
                }
                    
                for (int i=0; i<vx_num; i++){
                    ofs.write(( char * ) &vx_count[i],sizeof(int) );
                    // printf("count %d, cv %f\n", i, cvs[i]);
                    }//for
                ofs.close();

            CHECK(cudaFree(d_x));
            CHECK(cudaFree(d_y));
            CHECK(cudaFree(d_z));
            CHECK(cudaFree(d_vx_count));

            // free(x);
            // free(y);
            // free(z);
            free(vx_count);
            // free(vx_count_ratio);
// free(spc);

        }
    }
   }
}
   //end timer
    double iElaps = cpuSecond() - iStart;
    printf("elapsed %f sec\n", iElaps);
}






// 

