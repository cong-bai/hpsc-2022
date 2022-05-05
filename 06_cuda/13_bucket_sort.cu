#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucketGather(int *key, int *bucket, int n, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  extern __shared__ int localSum[];
  for(int i = idx; i < n; i += stride) {
    atomicAdd(&localSum[key[i]], 1);
  }
  if (threadIdx.x == 0) {
    for (int i = 0; i < range; i++) {
      atomicAdd(&bucket[i], localSum[i]);
    }
  }
}

__global__ void prefixSum(int *a, int *temp, int n) {
  // The original scan() works well when n == blockdim.x,
  // but sometimes may fail when they don't equal
  for (int stride = 1; stride < n; stride <<= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      temp[i] = a[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      if (i >= stride) a[i] += temp[i - stride];
    }
    __syncthreads();
  }
}

__global__ void bucketFill(int *key, int *bucket, int n, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(int j = idx; j < bucket[0]; j += stride) {
    key[j] = 0;
  }
  for (int i = 1; i < range; i++) {
    for(int j = bucket[i - 1] + idx; j < bucket[i]; j += stride) {
      key[j] = i;
    }
  }
}

int main() {
  int n = 10000;
  int range = 2000;
  const int THREAD_PER_BLOCK = 256;
  const int BLOCK_NUM = 16;
  int *key, *bucket, *temp;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&temp, range * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucketGather<<<BLOCK_NUM, THREAD_PER_BLOCK, range * sizeof(int)>>>(key, bucket, n, range);
  cudaDeviceSynchronize();
  // __syncthreads() syncs inside block
  // It seems hard to sync between blocks, so only 1 block is used
  prefixSum<<<1, THREAD_PER_BLOCK>>>(bucket, temp, range);
  cudaDeviceSynchronize();
  bucketFill<<<BLOCK_NUM, THREAD_PER_BLOCK>>>(key, bucket, n, range);
  cudaDeviceSynchronize();

  // std::vector<int> bucket(range); 
  // for (int i=0; i<range; i++) {
  //   bucket[i] = 0;
  // }
  // for (int i=0; i<n; i++) {
  //   bucket[key[i]]++;
  // }
  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(temp);
}
