/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "pngio.h"

#define FILTER_SIZE (3u)
#define TILE_SIZE (14u) // BLOCK_SIZE - 2 * (FILTER_SIZE / 2)
#define BLOCK_SIZE (16u)

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

double SMUDGE_MASK[] = 
{
	0.111, 0.111, 0.111,
	0.111, 0.111, 0.111,
	0.111, 0.111, 0.111,
};

double EDGE_MASK[] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1,
};

__global__ void processImage( unsigned char * out, const unsigned char * __restrict__ in, size_t pitch, unsigned int width, unsigned int height, 
const double* __restrict__  mask)
{
	int x_o = TILE_SIZE * blockIdx.x + threadIdx.x;
	int y_o = TILE_SIZE * blockIdx.y + threadIdx.y;
	int x_i = x_o - FILTER_SIZE / 2;
	int y_i = y_o - FILTER_SIZE / 2;
	unsigned int sum = 0;

	__shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];
	
	if( (x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height) )
		sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
	else
		sBuffer[threadIdx.y][threadIdx.x] = 0;
	
	__syncthreads();

	if( threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE ) {
		for( int r = 0; r < FILTER_SIZE; ++r )
			for( int c = 0; c < FILTER_SIZE; ++c )
				sum += (sBuffer[threadIdx.y + r][threadIdx.x + c]) * (mask[FILTER_SIZE * r + c]);
	
		if( x_o < width && y_o < height ) {
			//bsum /= FILTER_SIZE * FILTER_SIZE;
			out[ y_o * width + x_o ] = sum;
		}
	}
}

void printHelp()
{
	std::cout<<"Tento program pomoci konvoluce but rozmazava obraz, nebo provadi detekci hran."<<std::endl;
	std::cout<<"Parametry:"<<std::endl;
	std::cout<<"-h, --help vypise tuto napovedu."<<std::endl;
	std::cout<<"-r, --rozmazat provede rozmazani obrazku."<<std::endl;
	std::cout<<"-z, --zvyrazneni-hran zvyrazni hrany."<<std::endl;
}

using namespace std;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {
	
	const double* h_mask;
	
	if(argc == 2)
	{
		if(strcmp(argv[1],"-r")==0 || strcmp(argv[1],"--rozmazat")==0)
		{
			h_mask = SMUDGE_MASK;
		}
		else if (strcmp(argv[1],"-z") == 0 || strcmp(argv[1],"--zvyrazneni-hran") == 0)
		{
			h_mask = EDGE_MASK;
		}
		else if (strcmp(argv[1],"-h") == 0 || strcmp(argv[1],"--help") == 0)
		{
			printHelp();
			return EXIT_SUCCESS;
		}
		else
		{
			cerr<<"Neplatny prepinac! Vypisuji napovedu."<<endl;
			printHelp();
			return EXIT_FAILURE;
		}
	}
	else
	{
		cerr<<"Nespravny pocet parametru! Vypisuji napovedu."<<endl;
		printHelp();
		return EXIT_FAILURE;
	}
	
	printf("Processing image. See 'lena_new.png' image for the results...\n");
	
	/* Load image from file */
	png::image<png::rgb_pixel> img("../../resources/lena.png");
	
	unsigned int width = img.get_width();
	unsigned int height = img.get_height();
	
	/* Allocate memory buffers for the image processing */
	int size = width * height * sizeof(unsigned char);
	
	/* Allocate image buffers on the host memory */
	unsigned char *h_r = new unsigned char[ size ];
	unsigned char *h_g = new unsigned char[ size ];
	unsigned char *h_b = new unsigned char[ size ];
	
	unsigned char *h_r_n = new unsigned char[ size ];
	unsigned char *h_g_n = new unsigned char[ size ];
	unsigned char *h_b_n = new unsigned char[ size ];
	
	/* Convert PNG image to raw buffer */
	pvg::pngToRgb3( h_r, h_g, h_b, img );
	
	/* Allocate image buffre on GPGPU */
	unsigned char *d_r = NULL;
	unsigned char *d_g = NULL;
	unsigned char *d_b = NULL;
	size_t pitch_r = 0;
	size_t pitch_g = 0;
	size_t pitch_b = 0;
	
	unsigned char *d_r_n = NULL;
	unsigned char *d_g_n = NULL;
	unsigned char *d_b_n = NULL;
	
	CUDA_CHECK_RETURN( cudaMallocPitch( &d_r, &pitch_r, width, height ) );
	CUDA_CHECK_RETURN( cudaMallocPitch( &d_g, &pitch_g, width, height ) );
	CUDA_CHECK_RETURN( cudaMallocPitch( &d_b, &pitch_b, width, height ) );
	
	CUDA_CHECK_RETURN( cudaMalloc( &d_r_n, size ) );
	CUDA_CHECK_RETURN( cudaMalloc( &d_g_n, size ) );
	CUDA_CHECK_RETURN( cudaMalloc( &d_b_n, size ) );	
	
	/* Copy raw buffer from host memory to device memory */
	CUDA_CHECK_RETURN( cudaMemcpy2D( d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy2D( d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy2D( d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice) );
	
	double* d_SMUDGE_MASK = NULL;
	CUDA_CHECK_RETURN( cudaMalloc( &(d_SMUDGE_MASK), FILTER_SIZE * FILTER_SIZE * sizeof(double) ) );

    // Copy data to device memmory
    CUDA_CHECK_RETURN( cudaMemcpy( d_SMUDGE_MASK, h_mask, FILTER_SIZE * FILTER_SIZE * sizeof(double), cudaMemcpyHostToDevice) );
	
	
	/* Configure image kernel */
	dim3 grid_size( (width + TILE_SIZE - 1) / TILE_SIZE,
					(height + TILE_SIZE - 1) / TILE_SIZE );
					
	dim3 block_size( BLOCK_SIZE, BLOCK_SIZE );
	
	/* Run kernel and measure processing time */
	double start = omp_get_wtime();
	processImage<<< grid_size, block_size >>>( d_r_n, d_r, pitch_r, width, height, d_SMUDGE_MASK );
	processImage<<< grid_size, block_size >>>( d_g_n, d_g, pitch_g, width, height, d_SMUDGE_MASK );
	processImage<<< grid_size, block_size >>>( d_b_n, d_b, pitch_b, width, height, d_SMUDGE_MASK );
	/* Wait untile the kernel exits */
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	double end = omp_get_wtime();
	
	/* Copy raw buffer from device memory to host memory */
	CUDA_CHECK_RETURN( cudaMemcpy( h_r_n, d_r_n, size, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaMemcpy( h_g_n, d_g_n, size, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaMemcpy( h_b_n, d_b_n, size, cudaMemcpyDeviceToHost ) );
	
	/* Convert raw buffer to PNG image */
	pvg::rgb3ToPng( img, h_r_n, h_g_n, h_b_n );
	
	std::cout << "Done in " << end - start << " seconds." << std::endl;
	
	/* Write modified image to the disk */
	img.write("../lena_new.png");
	
	/* Free allocated buffers */
	CUDA_CHECK_RETURN( cudaFree( d_r ) );
	CUDA_CHECK_RETURN( cudaFree( d_r_n ) );
	
	CUDA_CHECK_RETURN( cudaFree( d_g ) );
	CUDA_CHECK_RETURN( cudaFree( d_g_n ) );

	CUDA_CHECK_RETURN( cudaFree( d_b ) );
	CUDA_CHECK_RETURN( cudaFree( d_b_n ) );

	delete [] h_r;
	delete [] h_r_n;

	delete [] h_g;
	delete [] h_g_n;

	delete [] h_b;
	delete [] h_b_n;
	
	return 0;
}