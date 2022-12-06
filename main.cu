#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "common.cuh"
#include "sha3.cuh"

#define BLOCKS 16
#define THREADS 256

#define N 1000000000

__device__ uint8_t dev_wanted_signature[4] = {0x0, 0x0, 0x0, 0x0};

__global__ void init_signature(uint32_t *fn_sig) {
    dev_wanted_signature[0] = *fn_sig >> 24;
    dev_wanted_signature[1] = ((*fn_sig >> 16) & 0xff);
    dev_wanted_signature[2] = ((*fn_sig >> 8) & 0xff);
    dev_wanted_signature[3] = ((*fn_sig >> 0) & 0xff);
}

__global__ void calculate(char *fn_name, char *fn_args) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init((unsigned long long)clock() + tid, 0, 0, &state);
    int block = (int)(curand_uniform_double(&state) * 1000000);

    int index = 0;
    char id[16];
    char data[128];

    memset(data, 0, 128);
    memcpy(data, fn_name, _strlen(fn_name));
    _itoa(block, id, 10);
    memcpy(&data[_strlen(data)], id, _strlen(id));
    memcpy(&data[_strlen(data)], "000000000", 9);
    index = _strlen(data);
    memcpy(&data[_strlen(data)], fn_args, _strlen(fn_args));

    for (int i = 0; i < N; i++) {
	_itoa(i, id, 10);
	memcpy(&data[index - _strlen(id)], id, _strlen(id));

	uint8_t hash[64];
	sha3_return_t ok = sha3_HashBuffer(256, SHA3_FLAGS_KECCAK, data,
					   _strlen(data), hash, 64);
	if (ok != 0) {
	    printf("bad params\n");
	    return;
	}

	if (hash[0] == dev_wanted_signature[0] &&
	    hash[1] == dev_wanted_signature[1] &&
	    hash[2] == dev_wanted_signature[2] &&
	    hash[3] == dev_wanted_signature[3]) {
	    printf(
		"thread => %d method => %s method id => 0x%02x%02x%02x%02x\n",
		tid, data, hash[0], hash[1], hash[2], hash[3]);
	}
    }
}

// TODO add speed
void *metrics(void *data) {
    char spin[4] = {'-', '\\', '|', '/'};

    while (1) {
	for (int i = 0; i < 1000000; i++) {
	    usleep(100000);
	    printf("\33[2K\r Searching %c ", spin[i % 4]);
	}
    }
    return NULL;
}

int main(int argc, char **argv) {
    int opt;
    char *fvalue = NULL;  // function name
    char *avalue = NULL;  // arguments
    char *svalue = NULL;  // signature
    uint32_t signature;

    while ((opt = getopt(argc, argv, "f:a:s:")) != -1) {
	switch (opt) {
	    case 'f':
		fvalue = optarg;
		break;
	    case 'a':
		avalue = optarg;
		break;
	    case 's':
		svalue = optarg;
		if (svalue[0] != '0' || svalue[1] != 'x' ||
		    _strlen(svalue) != 10) {
		    fprintf(stderr, "Wrong signature format!\n");
		    exit(EXIT_FAILURE);
		}

		signature = strtoul(svalue + 2, NULL, 16);
		break;
	    default: /* '?' */
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
    }

    if (fvalue == NULL || avalue == NULL || svalue == NULL) {
	usage(argv[0]);
	exit(EXIT_FAILURE);
    }

    pthread_t th;
    int ret = pthread_create(&th, NULL, &metrics, NULL);
    if (ret != 0) {
	printf("Error: pthread_create() failed\n");
	return 1;
    }

    // device arguments
    char *dev_f, *dev_a;
    uint32_t *dev_s;

    HANDLE_ERROR(cudaMalloc((void **)&dev_f, _strlen(fvalue) * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, _strlen(avalue) * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_s, sizeof(uint32_t)));

    HANDLE_ERROR(cudaMemcpy(dev_f, fvalue, _strlen(fvalue) * sizeof(char),
			    cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_a, avalue, _strlen(avalue) * sizeof(char),
			    cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_s, &signature, sizeof(uint32_t),
			    cudaMemcpyHostToDevice));

    init_signature<<<1, 1>>>(dev_s);
    HANDLE_ERROR(cudaFree(dev_s));

    calculate<<<BLOCKS, THREADS>>>(dev_f, dev_a);
    cudaDeviceSynchronize();  // not important

    HANDLE_ERROR(cudaFree(dev_f));
    HANDLE_ERROR(cudaFree(dev_a));
    exit(EXIT_SUCCESS);
}
