/*Compile: nvcc -o cuda filtercuba.cu*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)
#define BLOCK_SIZE  24

unsigned char* openGreyImage(char *path, unsigned int height, unsigned int width);
unsigned char** openRGBImage(char *path, unsigned int height, unsigned int width);
void saveGreyImage(char *path, unsigned char *image_array, unsigned int height, unsigned int width);
void saveRGBImage(char *path, unsigned char **image_array, unsigned int height, unsigned int width);
__global__ void convolution(unsigned char *in_image, unsigned char *out_image, int height, int width, int *cfilter);

/*
 * Open the images, process it and save it
 */
int main(int argc, char **argv) {
	int i, j, n, is_RGB, filter[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1}, *cfilter;
	unsigned int width, height;
	float elapsed_time;
	char *input_path, *output_path;
	unsigned char *Grey_image_array, **RGB_image_array, *tempGrey, **tempRGB, *in_image, *out_image;
	cudaEvent_t start, stop;	

	//if arguments is OK
	if (argc < 5 || argc > 6) {
		fprintf(stderr, "Not correct arguments. Please give me 1) the image file path, 2) the width and 3) the height of image, 4) 0 or 1 if it's Grey or RGB and 5) optinal the number of times that the filter will be applied.\n");
		exit(EXIT_FAILURE);
	}

	input_path = argv[1];
	width = atoi(argv[2]);
	height = atoi(argv[3]);
	is_RGB = atoi(argv[4]);
	if(argc == 6)
		n = atoi(argv[5]);
	else
		n = -1;

	//open the image
	if(!is_RGB)
		Grey_image_array = openGreyImage(input_path, height, width);
	else
		RGB_image_array = openRGBImage(input_path, height, width);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);
	dim3 dimGrid(FRACTION_CEILING(width, BLOCK_SIZE),FRACTION_CEILING(height, BLOCK_SIZE),1);

	cudaMalloc((void**)&in_image, height * width * sizeof(unsigned char));
	cudaMalloc((void**)&out_image, height * width * sizeof(unsigned char));
	cudaMalloc((void**)&cfilter, 9 * sizeof(int));

	cudaMemcpy(cfilter, filter, 9 * sizeof(int), cudaMemcpyHostToDevice);
	
	//time starts
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	if(n < 0){
		//repeat till the two array is equal, use memcpy so that cpu compare the arrays
		i = 0;
		if(!is_RGB){
			tempGrey = (unsigned char*)malloc(height * width * sizeof(unsigned char));
			do{
				if(i % 2 == 0){
					cudaMemcpy(in_image, Grey_image_array, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
					convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
					cudaThreadSynchronize();
					cudaMemcpy(tempGrey, out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				}else{
					cudaMemcpy(in_image, tempGrey, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
					convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
					cudaThreadSynchronize();
					cudaMemcpy(Grey_image_array, out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				}
				i++;
			}while(memcmp(Grey_image_array, tempGrey, height * width) != 0);
			free(tempGrey);
		}else{
			tempRGB = (unsigned char**)malloc(3 * sizeof(unsigned char*));
			for(j = 0; j < 3; j++)
				tempRGB[j] = (unsigned char*)malloc(height * width * sizeof(unsigned char));
			do{
				if(i % 2 == 0){
					for(j = 0; j < 3; j++){
						cudaMemcpy(in_image, RGB_image_array[j], height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
						convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
						cudaThreadSynchronize();
						cudaMemcpy(tempRGB[j], out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					}
				}else{
					for(j = 0; j < 3; j++){
						cudaMemcpy(in_image, tempRGB[j], height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
						convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
						cudaThreadSynchronize();
						cudaMemcpy(RGB_image_array[j], out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					}
				}
				i++;
			}while(memcmp(RGB_image_array[0], tempRGB[0], height * width) != 0 || memcmp(RGB_image_array[1], tempRGB[1], height * width) != 0 || memcmp(RGB_image_array[2], tempRGB[2], height * width) != 0);
			for(j = 0; j < 3; j++)
				free(tempRGB[j]);
			free(tempRGB);
		}
	}else{
		//repeat n times, use swap because no need to swap
		if(!is_RGB){
			cudaMalloc((void**)&tempGrey, height * width * sizeof(unsigned char));
			cudaMemcpy(in_image, Grey_image_array, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
			for(i = 0; i < n; i++){
				convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
				cudaThreadSynchronize();
				tempGrey = in_image;
				in_image = out_image;
				out_image = tempGrey;
			}
			cudaMemcpy(Grey_image_array, out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			cudaFree(tempGrey);
		}else{
			tempRGB = (unsigned char**)malloc(3 * sizeof(unsigned char*));
			for(i = 0; i < 3; i++)
				cudaMalloc((void**)&tempRGB[i], height * width * sizeof(unsigned char));
			for(i = 0; i < 3; i++){
				cudaMemcpy(in_image, RGB_image_array[i], height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
				for(j = 0; j < n; j++){
					convolution<<<dimGrid,dimBlock>>>(in_image, out_image, height, width, cfilter);
					cudaThreadSynchronize();
					tempRGB[i] = in_image;
					in_image = out_image;
					out_image = tempRGB[i];
				}
				cudaMemcpy(RGB_image_array[i], out_image, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			}
			for(i = 0; i < 3; i++)
				cudaFree(tempRGB[i]);
			free(tempRGB);
		}
	}

	//time finishes
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsed_time, start, stop);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	printf("Finished for file %s after %f sec\n", input_path, elapsed_time/1000);
	
	//append "(convolution)" at the end of the filename
	output_path =  (char*) malloc((strlen(input_path) + 1 + strlen("(convolution)")) * sizeof(char));
	strncpy(output_path, input_path, strlen(input_path) - strlen(".raw"));
	strcat(output_path, "(convolution)");
	strcat(output_path, ".raw");
	
	//save image in another file
	if(!is_RGB)
		saveGreyImage(output_path, Grey_image_array, height, width);
	else
		saveRGBImage(output_path, RGB_image_array, height, width);

	//time to leave, destroy dynamic space
	if(!is_RGB)
		free(Grey_image_array);
	else{
		for(i = 0; i < 3; i++)
			free(RGB_image_array[i]);
		free(RGB_image_array);
	}
	cudaFree(in_image);
	cudaFree(out_image);
	cudaFree(cfilter);

	//bye
	exit(EXIT_SUCCESS);
}


/*
 * Reads the Grey image file and store it into an array
 */
unsigned char* openGreyImage(char *path, unsigned int height, unsigned int width) {

	unsigned char *image_array = (unsigned char*) malloc(height * width * sizeof(unsigned char));

	FILE* image_file = fopen(path, "r");
	if (image_file == NULL) {
		perror("fopen failed");
		exit(EXIT_FAILURE);
	}

	if (fread(image_array, sizeof(unsigned char), height * width, image_file) != height * width) {
		fprintf(stderr, "fread failed");
		exit(EXIT_FAILURE);
	}

	fclose(image_file);

	return image_array;
}

/*
 * Reads the RGB image file and store each colour to an array
 */
unsigned char** openRGBImage(char *path, unsigned int height, unsigned int width){

	unsigned char *temp = (unsigned char*) malloc(3 * height * width * sizeof(unsigned char));

	FILE* image_file = fopen(path, "r");
	if (image_file == NULL) {
		perror("fopen failed");
		exit(EXIT_FAILURE);
	}

	if (fread(temp, sizeof(unsigned char), 3 * height * width, image_file) !=3* height * width) {
		fprintf(stderr, "fread failed");
		exit(EXIT_FAILURE);
	}

	fclose(image_file);

	unsigned int i;
	unsigned char **image_array = (unsigned char**) malloc(3 * sizeof(unsigned char*));
	for (i = 0; i < 3; i++){
		image_array[i] = (unsigned char*) malloc(height * width * sizeof(unsigned char));
	}
	for (i = 0; i < height * width * 3; i++){
		image_array[i % 3][i / 3] = temp[i];
	}
	free(temp);
	return image_array;
}

/*
 * Save Grey image to another image file specified by path
 */
void saveGreyImage(char *path, unsigned char *image_array, unsigned int height, unsigned int width) {
	
	FILE* image_file = fopen(path, "w");
	if (image_file == NULL) {
		perror("fopen failed");
		exit(EXIT_FAILURE);
	}

	if (fwrite(image_array, sizeof(unsigned char), height * width, image_file) != height * width) {
		fprintf(stderr, "fwrite failed");
		exit(EXIT_FAILURE);
	}

	fclose(image_file);

	return;

}

/*
 * Save RGB image to another image file specified by path
 */
void saveRGBImage(char *path, unsigned char **image_array, unsigned int height, unsigned int width){

	unsigned int i;
	unsigned char *temp = (unsigned char*) malloc(3 * height * width * sizeof(unsigned char));

	for (i = 0; i < height * width * 3; i++){
		temp[i] = image_array[i % 3][i / 3];
	}

	FILE* image_file = fopen(path, "w");
	if (image_file == NULL) {
		perror("fopen failed");
		exit(EXIT_FAILURE);
	}

	if (fwrite(temp, sizeof(unsigned char), height * width*3, image_file) != height * width*3) {
		fprintf(stderr, "fwrite failed");
		exit(EXIT_FAILURE);
	}

	fclose(image_file);

	free(temp);

	return;
}

/*
 * Function for gpu which apply the filter
 */
__global__ void convolution(unsigned char *in_image, unsigned char *out_image, int height, int width, int *cfilter ) {
	
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y >= height || x >= width){
    	   return;
	}
    	

	int i, j, s = 1, y_idx, x_idx, sum = 0;

	for (i = -s; i <= s; i++) {
		for ( j = -s; j <= s; j++) {
			y_idx = y + i;
			x_idx = x + j;
			if (y_idx >= height || y_idx < 0 || x_idx >= width || x_idx < 0) {
				y_idx = y;
				x_idx = x;
			}
			sum += in_image[width*(y_idx)+(x_idx)] * cfilter[3*(i+1)+(j+1)];
		}	
	}

	out_image[width*y+x] =(unsigned char)((float)sum/16);	
}

