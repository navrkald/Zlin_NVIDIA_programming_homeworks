#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
#include <fstream>

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

#define BLOCK_SIZE (64L)

typedef struct
{
    unsigned int width;
    unsigned int height;
    unsigned int numOfElements;
    long* data;
}TMatrix;


__global__ void graphicCardMatrixMultiplication(TMatrix h_matrix_A, TMatrix h_matrix_B, TMatrix matrix_result)
{
    long temp = 0;
    unsigned int rowidx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int columnidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(columnidx < matrix_result.width && rowidx < matrix_result.height)
    {
        for (unsigned int j = 0; j < h_matrix_A.width; j++)
        {
            temp += (h_matrix_A.data)[h_matrix_A.width * rowidx + j] * (h_matrix_B.data)[ h_matrix_B.width * j + columnidx];
        }    
        (matrix_result.data)[ columnidx + matrix_result.width * rowidx] = temp;
    }
}

using namespace std;

void printMatrix (TMatrix matrix)
{
    for(unsigned int j = 0; j < matrix.height; j++){
        for(unsigned int k=0; k < matrix.width; k++){
            printf("%ld ",(matrix.data)[ j * matrix.width + k]);
        }
        printf("\n");
    }
    printf("\n");
}

int readMatrixFromFile(TMatrix* matrix, const char* fileName)
{
	#ifdef DEBUG
		cout<<"DEBUG: Opening "<<fileName<<"."<<endl;
	#endif
    ifstream fileStream(fileName);
    if(fileStream.is_open())
    {
    	#ifdef DEBUG
    		cout<<"DEBUG: File "<<fileName<<" was sucessfully open."<<endl;
    	#endif
    	
		
		if(!(fileStream >> matrix->width)) 
		{
		    cerr<<"ERROR: Load matrix width failed from file "<<fileName<<endl;
		    return EXIT_FAILURE;
		}
		if(!(fileStream >> matrix->height))
		{
		    cerr<<"ERROR: Load matrix width failed from file "<<fileName<<endl;
		    return EXIT_FAILURE;
		}
		
		#ifdef DEBUG
    		cout<<"DEBUG: Matrix width is "<<matrix.width<<" and height is "<<matrix.height<<endl;
    	#endif
		
		matrix->numOfElements = matrix->width * matrix->height;
		
		if(matrix->numOfElements == 0)
		{
		    cerr<<"ERROR: width or height is zero."<<endl;
		    return EXIT_FAILURE;
		}
		
		matrix->data = new long[matrix->numOfElements];
		long loadedNumber;
		unsigned int idx = 0;
		while(fileStream >> loadedNumber)
		{
			if(idx >= matrix->numOfElements)
			{
				cerr<<"WARNING: In file "<<fileName<<" are more numbers than the size of matrix."<<endl;
				break;
			}
			(matrix->data)[idx] = loadedNumber;
			idx++;
			#ifdef DEBUG
    			cout<<"DEBUG: Reading number "<<loadedNumber<<" form "<<fileName<<endl;
    		#endif
		}
	    
	    if((idx) != matrix->numOfElements)
	    {
	        cerr<<"ERROR: In file "<<fileName<<" size you specified is "<<matrix->numOfElements<< 
	        " but number of loaded numbers is "<<idx<<endl;
	    	free(matrix->data);
	    	return EXIT_FAILURE;
	    }
    }
    else
    {
    	cerr<<"Unable to open file: "<<fileName<<endl;
    	return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void printHelp()
{
    cout<<"This is help for program which counts 2 matrixes."<<endl;
    cout<<"These 2 matrixes has to be placed to 2 files, which are"<<endl;
    cout<<"specified each in files, which are passed as params to this program."<<endl;
    cout<<"On the first line has to be 2 not zero numebers which specifies width and height"<<endl;
    cout<<"of the matrixes."<<endl;
    cout<<"On the next lines has to be speficied data of the matrix."<<endl;
    cout<<"The numebes has to be whole numebers positive or negative."<<endl;
}

int main(int argc, char **argv) 
{
    const char* fileName1 = "";
    const char* fileName2 = "";
    if(argc == 2 && (strcmp(argv[1],"-h") == 0 || strcmp(argv[1],"--help") == 0 ))
    {
        printHelp();
        return EXIT_SUCCESS;
    }
    else if(argc == 3)
    {
        fileName1 = argv[1];
        fileName2 = argv[2];
    }
    else
    {
        cerr<<"ERROR: Not correct parameters, now help will be printed."<<endl;
        printHelp();
        return EXIT_FAILURE;
    }

    TMatrix h_matrix_A, h_matrix_B, h_matrixResult;
    if(readMatrixFromFile(&h_matrix_A, fileName1) != EXIT_SUCCESS) 
    {
        return EXIT_FAILURE;
    }
    if(readMatrixFromFile(&h_matrix_B, fileName2) != EXIT_SUCCESS) 
    {
        return EXIT_FAILURE;
    }
    
    if(h_matrix_A.width != h_matrix_B.height)
    {
        cerr<<"ERROR: To able multiply two matrixes is neede to be equal frist widht seconds height"<<endl;
        free(h_matrix_A.data);
        free(h_matrix_B.data);
        return EXIT_FAILURE;
    }
    
    // Prepare matrix size
    h_matrixResult.height = h_matrix_A.height;
    h_matrixResult.width = h_matrix_B.width;
    h_matrixResult.numOfElements = h_matrixResult.height * h_matrixResult.width;
    
    // Device matrixes (in graphic card memmory)
    TMatrix d_matrix_A, d_matrix_B, d_matrixResult;
    
    //Set sizes of matrixes in device memmory
    d_matrix_A.width = h_matrix_A.width;
    d_matrix_A.height = h_matrix_A.height;
    d_matrix_B.width = h_matrix_B.width;
    d_matrix_B.height = h_matrix_B.height;
    d_matrixResult.width = h_matrixResult.width;
    d_matrixResult.height = h_matrixResult.height;
    
    // Alocate device memmory
    CUDA_CHECK_RETURN( cudaMalloc( &(d_matrix_A.data), h_matrix_A.width * h_matrix_A.height * sizeof(long) ) );
	CUDA_CHECK_RETURN( cudaMalloc( &(d_matrix_B.data), h_matrix_B.width * h_matrix_B.height * sizeof(long) ) );
    CUDA_CHECK_RETURN( cudaMalloc( &(d_matrixResult.data), h_matrixResult.width * h_matrixResult.height * sizeof(long) ) );

    // Copy data to device memmory
    CUDA_CHECK_RETURN( cudaMemcpy( d_matrix_A.data, h_matrix_A.data, h_matrix_A.height * h_matrix_A.width * sizeof(long), cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy( d_matrix_B.data, h_matrix_B.data, h_matrix_B.height * h_matrix_B.width * sizeof(long), cudaMemcpyHostToDevice) );

    // Prepare kernel
	dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE); // Konfigurace kernelu
    dim3 dimGrid((h_matrixResult.height * h_matrixResult.width + BLOCK_SIZE - 1 ) / BLOCK_SIZE, (h_matrixResult.height * h_matrixResult.width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Multiply on device
    graphicCardMatrixMultiplication<<< dimBlock, dimGrid >>>(d_matrix_A, d_matrix_B, d_matrixResult);
    
    // Wait for finish counting
    CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
    
    // Alocate memmory in ram and copy data from device to ram
    h_matrixResult.data = new long[h_matrixResult.numOfElements];
	CUDA_CHECK_RETURN(cudaMemcpy(h_matrixResult.data, d_matrixResult.data, h_matrixResult.height * h_matrixResult.width * sizeof(long), cudaMemcpyDeviceToHost));
    
    // Write results
    cout<<"Matrix A equals:"<<endl;
    printMatrix (h_matrix_A);
    cout<<"Matrix B equals:"<<endl;
    printMatrix (h_matrix_B);
    cout<<"A * B equals:"<<endl;
    printMatrix (h_matrixResult);
    
    // Free memmory in device
    CUDA_CHECK_RETURN( cudaFree( d_matrix_A.data ));
	CUDA_CHECK_RETURN( cudaFree( d_matrix_B.data ));
	CUDA_CHECK_RETURN( cudaFree( d_matrixResult.data ));

    // Free memmory in ram
	free( h_matrix_A.data );
	free( h_matrix_B.data );
	free( h_matrixResult.data );

    return EXIT_SUCCESS;
}