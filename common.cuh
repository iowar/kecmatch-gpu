#ifndef __COMMON_H__
#define __COMMON_H__
#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
	printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
	exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

extern void usage(char *arg);

extern __host__ __device__ size_t _strlen(char s[]);
extern __device__ int _itoa(int value, char *sp, int radix);

#endif
