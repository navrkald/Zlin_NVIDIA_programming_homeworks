/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <climits>


// For testing of performance of GPU and CPU
// When this is defined only times are counted
// #define TEST_PERFORMANCE  

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }

#define BLOCK_SIZE (128L)

__global__ void vectFill( int * data1, int * data2, int * restult, unsigned long sizeOfArray )
{
	unsigned long i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i < sizeOfArray )
	{
		restult[ i ] = data1[i] + data2[i];	
	}
}

void printArray(unsigned long numOfElements, int* array)
{
	printf("[");
	for(unsigned long i = 0; i < numOfElements; i++)
	{
		printf(" %d,",array[i]);
	}
	printf("\b]");
}

void printResults(unsigned long numOfElements, int* input1, int* input2, int* result, double escapedTime)
{
#ifndef TEST_PERFORMANCE	
	printf("A + B = C\n");
	printf("A = ");
	printArray(numOfElements, input1);
	printf("\n");
	printf("B = ");
	printArray(numOfElements, input2);
	printf("\n");
	printf("C = ");
	printArray(numOfElements, result);
	printf("\n");
#endif
	printf("Add two arrays of size %lu computed in %lf seconds\n", numOfElements, escapedTime);
}

void testPerformance(unsigned long elemCount)
{
	printf("============================================\n");
	double totalBytes = sizeof(int)*elemCount;
	const char* byte = "B";
	const char* kiloByte = "KB";
	const char* megaByte = "MB";
	const char* gigaByte = "GB";
	const char** unit = &byte;
	double giga = 1024*1024*1024;
	double mega = 1024*1024;
	double kilo = 1024;
	if(totalBytes >= giga)
	{
		totalBytes /=giga; 
		unit = &gigaByte;
	}
	else if (totalBytes >= mega)
	{
		totalBytes /= mega;
		unit = &megaByte;
	}
	else if(totalBytes >= kilo)
	{
		totalBytes /= kilo;
		unit = &kiloByte;
	}
	
	printf("Add two arrays each of size %lf %s\n", totalBytes, *unit);
	double startTime, endTime, start, end;
	start = omp_get_wtime();
	/* Allocate data buffer in host memory */
	startTime = omp_get_wtime();
	int *h_data1 = (int*) malloc( elemCount * sizeof(int) );
	int *h_data2 = (int*) malloc( elemCount * sizeof(int) );
	int *h_data3 = (int*) malloc( elemCount * sizeof(int) );
	for(unsigned long i = 0; i < elemCount; i++)
	{
		h_data1[i] = i;
		h_data2[i] = i;
	}
	
	memset( h_data3, 0, elemCount * sizeof(int) );
	endTime = omp_get_wtime();
	printf("Prepare data: \t\t%lf s\n", endTime - startTime);
	
	/* Allocate data buffer in device memory */
	int *d_data1 = NULL;
	int *d_data2 = NULL;
	int *d_data3 = NULL;
	
	startTime = omp_get_wtime();
	CUDA_CHECK_RETURN( cudaMalloc( &d_data1, elemCount * sizeof(int) ) );
	CUDA_CHECK_RETURN( cudaMalloc( &d_data2, elemCount * sizeof(int) ) );
	CUDA_CHECK_RETURN( cudaMalloc( &d_data3, elemCount * sizeof(int) ) );

	CUDA_CHECK_RETURN( cudaMemcpy( d_data1, h_data1, elemCount * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy( d_data2, h_data2, elemCount * sizeof(int), cudaMemcpyHostToDevice) );
	endTime = omp_get_wtime();
	printf("RAM -> graphic \t\t%lf s\n", endTime - startTime);
	
	/* Configure kernel */
	int blockSize = BLOCK_SIZE;
	int gridSize = (elemCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	/* Run kernel */
	startTime = omp_get_wtime();
	vectFill<<< gridSize, blockSize >>>( d_data1, d_data2, d_data3, elemCount);
	endTime = omp_get_wtime();
	printf("Kernel: \t\t%lf s\n", endTime - startTime);
	
	startTime = omp_get_wtime();
	/* Wait until the kernel finishes its work */
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	
	// Musim provadet?
	//CUDA_CHECK_RETURN( cudaMemcpy( h_data1, d_data1, elemCount * sizeof(int), cudaMemcpyDeviceToHost) );
	//CUDA_CHECK_RETURN( cudaMemcpy( h_data2, d_data2, elemCount * sizeof(int), cudaMemcpyDeviceToHost) );
	
	CUDA_CHECK_RETURN( cudaMemcpy( h_data3, d_data3, elemCount * sizeof(int), cudaMemcpyDeviceToHost) );
	
	printResults(elemCount, h_data1, h_data2, h_data3, 0.0);
	
	endTime = omp_get_wtime();
	printf("GPU -> RAM: \t\t%lf s\n", endTime - startTime);
	
	
	startTime = omp_get_wtime();
	CUDA_CHECK_RETURN( cudaFree( d_data1) );
	CUDA_CHECK_RETURN( cudaFree( d_data2) );
	CUDA_CHECK_RETURN( cudaFree( d_data3) );
	
	free( h_data1 );
	free( h_data2 );
	free( h_data3 );
	endTime = omp_get_wtime();
	printf("Free takes: \t\t%lf s\n", endTime - startTime);
	end = omp_get_wtime();
	printf("Total time: \t\t%lf s\n", end - start);
	return;
}

#define NUM_OF_ARRAYS 3
#define _4_GB 4000000000
#define _5_GB 5000000000

void add2vectorOnCpu(unsigned long elemCount)
{
	printf("========================\n");
	double startTime = omp_get_wtime();
	int *data1 = (int*) malloc( elemCount * sizeof(int) );
	int *data2 = (int*) malloc( elemCount * sizeof(int) );
	int *data3 = (int*) malloc( elemCount * sizeof(int) );
	for(unsigned long i = 0; i < elemCount; i++)
	{
		data1[i] = i;
		data2[i] = i;
	}
	for(unsigned long i = 0; i < elemCount; i++)
	{
		data3[i] = data1[i] + data2[i];
	}
	free(data1);
	free(data2);
	free(data3);
	double endTime = omp_get_wtime();
	printf("Counted 1GB on CPU takes: %lf s\n", endTime - startTime);
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) 
{
	
	testPerformance(2);
	testPerformance(8);
	testPerformance(16);
	#ifdef TEST_PERFORMANCE
		testPerformance(256);
		testPerformance(256*1024);
		testPerformance(256*1024*10);
		testPerformance(256*1024*100);
		testPerformance(256*1024*500);
		testPerformance(256*1024*800);
		testPerformance(256*1024*1024);
		add2vectorOnCpu(256*1024*1024);
	#endif
	
	
	// //size_t maxSize = _4_GB / (sizeof(int)*NUM_OF_ARRAYS);
	//size_t maxSize = _5_GB / (sizeof(int)*NUM_OF_ARRAYS); // -> Resulted that cuda was killed
	//testPerformance(UINT_MAX); // -> resulted to sefmentation fault
	//testPerformance(maxSize);
	return 0;
}