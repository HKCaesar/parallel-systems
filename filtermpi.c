#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "mpi.h"
#define SUM 16

MPI_Datatype coltype, rowtype;
const int filter[9] = {1,2,1,2,4,2,1,2,1};


unsigned char* getGreyImageArray(char *path, unsigned int height, unsigned int width);
void passGreyChangestoFile(char* path, void* image_array, unsigned int height, unsigned int width);
unsigned char** openRGBImage(char *path, unsigned int height, unsigned int width);
void saveRGBImage(char *path, unsigned char **image_array, unsigned int height, unsigned int width);
void apply_outer_filter(unsigned char** locbuf, unsigned int width, unsigned int height, unsigned char choose ,unsigned char* up,unsigned char*down,unsigned char* left,unsigned char* right,unsigned char ul,unsigned char ur,unsigned char dl, unsigned char dr);
void apply_inner_filter(unsigned char** locbuf,unsigned int width, unsigned int height,unsigned char choose);
void send_pixels(int myid, int numprocs, unsigned char* locbuf, unsigned int width, unsigned int height, int dim,MPI_Request* request,int tag);
void recv_pixels(int myid, int numprocs,unsigned char* ul, unsigned char* ur, unsigned char* dr, unsigned char* dl,unsigned char*  up, unsigned char* left, unsigned char* right, unsigned char* down, unsigned int width, unsigned int height, int dim,MPI_Request* request,int tag);

int main(int argc, char** argv) {
	int i, myid, numprocs, dim, *sendcounts, *displs, RGB, change, glob_conv,rep;
	if (argc != 6) {
		fprintf(stderr,
		"Not correct arguments. Please give me 1) the path of the image file, 2) the width, 3) the height 4)0 if image is grey 1 otherwise 5)repetitions if we don't reach convergence .\n");
		exit(EXIT_FAILURE);
	}
	char *path = argv[1];
	double start_time, end_time, elapsed_time;
	rep = atoi(argv[5]);	//number of iterations
	RGB=atoi(argv[4]); //0 for grey image
	unsigned int width = atoi(argv[2]);
	unsigned int height = atoi(argv[3]);	//Get the  dimensions

	MPI_Datatype blocktype, blocktype2;


	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

	dim= (int) sqrt(numprocs);	//square root

	MPI_Type_vector(height/dim, 1, width/dim, MPI_UNSIGNED_CHAR, &coltype); //(numblocks, blocksize, stride, oldtype, newtype);
	MPI_Type_commit(&coltype);

	MPI_Type_vector(1, width/dim, 1, MPI_UNSIGNED_CHAR, &rowtype); //(numblocks, blocksize, stride, oldtype, newtype);
	MPI_Type_commit(&rowtype);

	MPI_Type_vector(height/dim, width/dim, width, MPI_UNSIGNED_CHAR, &blocktype2); //(numblocks, blocksize, stride, oldtype, newtype);
	MPI_Type_create_resized( blocktype2, 0, sizeof(unsigned char), &blocktype);
	MPI_Type_commit(&blocktype);

	displs= (int*) malloc(numprocs*sizeof(int));	//needed in scatter and gather
	sendcounts= (int*) malloc(numprocs*sizeof(int));
	for (i=0; i< numprocs; i++){
		displs[i]= (i/dim)*width*height/dim + width/dim*(i%dim);
		sendcounts[i]=1;
	}

	if (!RGB){
		unsigned char* image_array, **locbuf, *up=NULL, *down=NULL, *right=NULL, *left=NULL,ul,dl,ur,dr,choose=0;
		MPI_Status recv_status[8], send_status[8];
		MPI_Request send_req[8], recv_req[8];
		if (myid == 0){
			image_array = getGreyImageArray(path, height, width); //read image
			if (image_array ==NULL){
				MPI_Abort(MPI_COMM_WORLD,1);
			}
			printf("image opened\n");
		}	//initialize arrays that receive pixels from other processes
		if (myid < numprocs - dim){ //set down
			down= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
		}
		if (myid /dim){ //set up
			up= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
		}
		if (myid % dim < dim-1){ //set right
			right= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
		}
		if (myid%dim){ //set left
			left= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
		}
		//initialize local buffer. It has size 2, so we store both the previous and the current image
		locbuf= (unsigned char**) malloc(2*sizeof(unsigned char*));
		locbuf[0]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbuf[1]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));

		for(i=0; i < 8; i++){
			send_req[i] = MPI_REQUEST_NULL;
			recv_req[i] = MPI_REQUEST_NULL;
		}

		if (myid ==0)	//start timing
			start_time = MPI_Wtime();
		//scatter image
		MPI_Scatterv(image_array, sendcounts, displs,blocktype, locbuf[0],width*height/numprocs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

		//the main part of communication
		for (i=0;i<rep;i++){
			send_pixels(myid, numprocs, locbuf[choose], width, height, dim, send_req,1); //send and receive

			recv_pixels(myid, numprocs, &ul, &ur, &dr, &dl, up, left, right, down, width, height, dim, recv_req,1);

			apply_inner_filter(locbuf, width/dim, height/dim, choose);	//exploit time until we have received data

			MPI_Waitall(8, recv_req, recv_status);	//wait to receive what we want

			apply_outer_filter(locbuf, width/dim, height/dim, choose,up,down, left, right, ul,ur,dl,dr);

			MPI_Waitall(8, send_req, send_status);	//wait to send
			choose=1-choose;

				if (memcmp(locbuf[0], locbuf[1], height * width/numprocs) ) //check for convergence
					change = 1;
				else
					change = 0;
				MPI_Allreduce(&change, &glob_conv, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD); //bitwise or for all changes 
				if (glob_conv)
					continue;
				printf("Converged!\n");
				rep =i+1;
				break;
		}
		MPI_Gatherv( locbuf[choose], width*height/numprocs,MPI_UNSIGNED_CHAR ,image_array , sendcounts, displs, blocktype, 0, MPI_COMM_WORLD);

		if (myid ==0){	//print time
			end_time = MPI_Wtime();
			elapsed_time = end_time - start_time;
			printf("total time for grey image %dx%d for %d repetitions is %f\n", width, height, rep, elapsed_time);
			passGreyChangestoFile("final_image", image_array, height, width);
		}
	if (left !=NULL)	//free memory
		free(left);
	if (up !=NULL)
		free(up);
	if (right !=NULL)
		free(right);
	if (down !=NULL)
		free(down);

	free(locbuf[0]);
	free(locbuf[1]);
	free(locbuf);
	}
	else{	//we follow exactly the same process as above but we do everything 3 times
		unsigned char** image_array, *temp=NULL,*temp1=NULL,*temp2=NULL,**locbufr,**locbufg,**locbufb, *upr=NULL, *downr=NULL, *rightr=NULL, *leftr=NULL,ulr,dlr,urr,drr,choose=0;
		unsigned char *upg=NULL, *downg=NULL, *rightg=NULL, *leftg=NULL,ulg,dlg,urg,drg,*upb=NULL, *downb=NULL, *rightb=NULL, *leftb=NULL,ulb,dlb,urb,drb;
		MPI_Status recv_statusr[8], send_statusr[8],recv_statusg[8], send_statusg[8],recv_statusb[8], send_statusb[8];
		MPI_Request send_reqr[8], recv_reqr[8],send_reqg[8], recv_reqg[8],send_reqb[8], recv_reqb[8];
		if (myid == 0){
			image_array = openRGBImage(path, height, width);
			if (image_array ==NULL){
				MPI_Abort(MPI_COMM_WORLD,1);
			}
			printf("image opened\n");
		}

		if (myid < numprocs - dim){ //set down
			downr= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
			downg= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
			downb= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
		}
		if (myid /dim){ //set up
			upr= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
			upg= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
			upb= (unsigned char*) malloc(width/dim*sizeof(unsigned char));
		}
		if (myid % dim < dim-1){ //set right
			rightr= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
			rightg= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
			rightb= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
		}
		if (myid%dim){ //set left
			leftr= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
			leftg= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
			leftb= (unsigned char*) malloc(height/dim*sizeof(unsigned char));
		}

		locbufr= (unsigned char**) malloc(2*sizeof(unsigned char*));
		locbufr[0]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbufr[1]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbufg= (unsigned char**) malloc(2*sizeof(unsigned char*));
		locbufg[0]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbufg[1]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbufb= (unsigned char**) malloc(2*sizeof(unsigned char*));
		locbufb[0]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));
		locbufb[1]= (unsigned char*) malloc(width*height/numprocs*sizeof(unsigned char));


		for(i=0; i < 8; i++){
			send_reqr[i] = MPI_REQUEST_NULL;
			recv_reqr[i] = MPI_REQUEST_NULL;
			send_reqg[i] = MPI_REQUEST_NULL;
			recv_reqg[i] = MPI_REQUEST_NULL;
			send_reqb[i] = MPI_REQUEST_NULL;
			recv_reqb[i] = MPI_REQUEST_NULL;
		}
	if (myid ==0){
		temp= image_array[0];
		temp1= image_array[1];
		temp2= image_array[2];
		start_time = MPI_Wtime();
	}
	MPI_Scatterv(temp, sendcounts, displs,blocktype, locbufr[0],width*height/numprocs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	    MPI_Scatterv(temp1, sendcounts, displs,blocktype, locbufg[0],width*height/numprocs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	    MPI_Scatterv(temp2, sendcounts, displs,blocktype, locbufb[0],width*height/numprocs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	for(i=0;i<rep;i++){
			send_pixels(myid, numprocs, locbufr[choose], width, height, dim, send_reqr,1);
			send_pixels(myid, numprocs, locbufg[choose], width, height, dim, send_reqg,10);
			send_pixels(myid, numprocs, locbufb[choose], width, height, dim, send_reqb,100);

			recv_pixels(myid, numprocs, &ulr, &urr, &drr, &dlr, upr, leftr, rightr, downr, width, height, dim, recv_reqr,1);
			recv_pixels(myid, numprocs, &ulg, &urg, &drg, &dlg, upg, leftg, rightg, downg, width, height, dim, recv_reqg,10);
			recv_pixels(myid, numprocs, &ulb, &urb, &drb, &dlb, upb, leftb, rightb, downb, width, height, dim, recv_reqb,100);

			apply_inner_filter(locbufr, width/dim, height/dim, choose);
			apply_inner_filter(locbufg, width/dim, height/dim, choose);
			apply_inner_filter(locbufb, width/dim, height/dim, choose);

			MPI_Waitall(8, recv_reqr, recv_statusr);
			apply_outer_filter(locbufr, width/dim, height/dim, choose,upr,downr, leftr, rightr, ulr,urr,dlr,drr);
			MPI_Waitall(8, recv_reqg, recv_statusg);
			apply_outer_filter(locbufg, width/dim, height/dim, choose,upg,downg, leftg, rightg, ulg,urg,dlg,drg);
			MPI_Waitall(8, recv_reqb, recv_statusb);
			apply_outer_filter(locbufb, width/dim, height/dim, choose,upb,downb, leftb, rightb, ulb,urb,dlb,drb);

			MPI_Waitall(8, send_reqr, send_statusr);
			MPI_Waitall(8, send_reqg, send_statusg);
			MPI_Waitall(8, send_reqb, send_statusb);
			choose=1-choose;
				if (memcmp(locbufr[0], locbufr[1], height * width/numprocs) )
					change = 1;
				else
					change = 0;
				MPI_Allreduce(&change, &glob_conv, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
				if (glob_conv)
					continue;

				if (memcmp(locbufg[0], locbufg[1], height * width/numprocs) )
					change = 1;
				else
					change = 0;
				MPI_Allreduce(&change, &glob_conv, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
				if (glob_conv)
					continue;

				if (memcmp(locbufb[0], locbufb[1], height * width/numprocs) )
					change = 1;
				else
					change = 0;
				MPI_Allreduce(&change, &glob_conv, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
				if (glob_conv)
					continue;
				printf("Converged!\n");
				rep =i+1;
				break;
	}

		MPI_Gatherv( locbufr[choose], width*height/numprocs,MPI_UNSIGNED_CHAR ,temp , sendcounts, displs, blocktype, 0, MPI_COMM_WORLD);
		MPI_Gatherv( locbufg[choose], width*height/numprocs,MPI_UNSIGNED_CHAR ,temp1 , sendcounts, displs, blocktype, 0, MPI_COMM_WORLD);
		MPI_Gatherv( locbufb[choose], width*height/numprocs,MPI_UNSIGNED_CHAR ,temp2 , sendcounts, displs, blocktype, 0, MPI_COMM_WORLD);


		if (myid ==0){
			end_time = MPI_Wtime();
			elapsed_time = end_time - start_time;
			printf("total time for rgb image %dx%d for %d repetitions is %f\n", width, height, rep, elapsed_time);
			saveRGBImage("final_image", image_array, height, width);
		}
	if (downr !=NULL){
		free(downr);
		free(downg);
		free(downb);
	}
	if (upr !=NULL){
		free(upr);
		free(upg);
		free(upb);
	}
	if (leftr !=NULL){
		free(leftr);
		free(leftg);
		free(leftb);
	}
	if (rightr !=NULL){
		free(rightr);
		free(rightg);
		free(rightb);
	}
	free(locbufr[0]);
	free(locbufr[1]);
	free(locbufr);
	free(locbufg[0]);
	free(locbufg[1]);
	free(locbufg);
	free(locbufb[0]);
	free(locbufb[1]);
	free(locbufb);
	}
	free(displs);
	free(sendcounts);
	MPI_Type_free(&blocktype2);
	MPI_Type_free(&blocktype);
	MPI_Type_free(&rowtype);
	MPI_Type_free(&coltype);
	MPI_Finalize();

	exit(EXIT_SUCCESS);
}

void send_pixels(int myid, int numprocs, unsigned char* locbuf, unsigned int width, unsigned int height, int dim,MPI_Request* request, int tag){

	if (myid /dim){	//send to the upper process
		MPI_Isend(&locbuf[0], 1,rowtype,myid-dim,5*tag,MPI_COMM_WORLD,&request[0]);
	}

	if (myid < numprocs - dim){	//send to the down process
		MPI_Isend(&locbuf[width/dim*(height/dim - 1)], 1,rowtype,myid + dim,1*tag,MPI_COMM_WORLD,&request[1]);
	}

	if (myid%dim){ //send to the left
		MPI_Isend(&locbuf[0], 1 ,coltype,myid -1,2*tag,MPI_COMM_WORLD,&request[2]);
	}

	if (myid % dim < dim-1){ //send to the right
		MPI_Isend(&locbuf[width/dim - 1], 1 ,coltype,myid +1,3*tag,MPI_COMM_WORLD,&request[3]);
	}

	if ( (myid%dim !=0) && (myid < numprocs - dim)){ //send to down left
		MPI_Isend(&locbuf[width/dim*(height/dim -1)], 1 ,MPI_UNSIGNED_CHAR,myid +dim -1,4*tag,MPI_COMM_WORLD,&request[4]);
	}

	if ( (myid%dim !=0) && (myid /dim)){ //send to upper left
		MPI_Isend(&locbuf[0], 1 ,MPI_UNSIGNED_CHAR,myid -dim -1,4*tag,MPI_COMM_WORLD,&request[5]);
	}

	if ( (myid%dim < dim-1) && (myid < numprocs - dim)){ //send to down right
		MPI_Isend(&locbuf[width/dim*height/dim -1], 1 ,MPI_UNSIGNED_CHAR,myid +dim +1,4*tag,MPI_COMM_WORLD,&request[6]);
	}


	if ( (myid%dim < dim-1) && (myid /dim)){ //send to upper right
		MPI_Isend(&locbuf[width/dim-1], 1 ,MPI_UNSIGNED_CHAR,myid -dim +1,4*tag,MPI_COMM_WORLD,&request[7]);
	}
}



void recv_pixels(int myid, int numprocs,unsigned char* ul, unsigned char* ur, unsigned char* dr, unsigned char* dl, unsigned char* up, unsigned char* left, unsigned char* right, unsigned char* down, unsigned int width, unsigned int height, int dim, MPI_Request* request, int tag){

	if (myid < numprocs - dim){ //recv from down process
		MPI_Irecv(down,1,rowtype ,myid+dim ,5*tag,MPI_COMM_WORLD,&request[0]);
	}

	if (myid /dim){ //recv from upper process
		MPI_Irecv(up,1,rowtype ,myid-dim ,1*tag,MPI_COMM_WORLD,&request[1]);
	}


	if (myid % dim < dim-1){ //recv from right process
		MPI_Irecv(right,height/dim, MPI_UNSIGNED_CHAR,myid+1 ,2*tag,MPI_COMM_WORLD,&request[2]);
	}

	if (myid%dim){ //recv from left process
		MPI_Irecv(left,height/dim, MPI_UNSIGNED_CHAR,myid-1 ,3*tag,MPI_COMM_WORLD,&request[3]);
	}

	if ((myid%dim < dim-1) && (myid /dim)){ //recv from upper right
		MPI_Irecv(ur,1, MPI_UNSIGNED_CHAR,myid-dim+1 ,4*tag,MPI_COMM_WORLD,&request[4]);
	}

	if ((myid%dim < dim-1) && (myid < numprocs - dim)){ //recv from down right
		MPI_Irecv(dr,1, MPI_UNSIGNED_CHAR,myid+dim+1 ,4*tag,MPI_COMM_WORLD,&request[5]);
	}

	if ((myid%dim !=0) && (myid /dim)){ //recv from upper left
		MPI_Irecv(ul,1, MPI_UNSIGNED_CHAR,myid-dim-1 ,4*tag,MPI_COMM_WORLD,&request[6]);
	}

	if ((myid%dim !=0) && (myid < numprocs - dim)){ //recv from down left
		MPI_Irecv(dl,1, MPI_UNSIGNED_CHAR,myid+dim-1 ,4*tag,MPI_COMM_WORLD,&request[7]);
	}
}

unsigned char calculate_value(unsigned char up0, unsigned char up1, unsigned char up2, unsigned char right, unsigned char me, unsigned char left, unsigned char d0, unsigned char d1, unsigned char d2){
	int temp = filter[0]*up0 + filter[1]*up1 +filter[2]*up2 + filter[3]*right + filter[4]*me + filter[5]*left +filter[6]*d0+ filter[7]*d1   + filter[8]*d2;
	return temp/SUM;
}

void apply_inner_filter(unsigned char** locbuf,unsigned int width, unsigned int height,unsigned char choose){
	int temp;
	unsigned int i;
	for (i=width; i< width*(height -1); i++){
		if ( ((i%width) == 0) || ((i%width) == (width -1)) )
			continue;

		locbuf[1-choose][i]= (filter[0]*locbuf[choose][i-width-1] + filter[1]*locbuf[choose][i-width]+ filter[2]*locbuf[choose][i-width+1] + filter[3]*locbuf[choose][i-1] + filter[4]*locbuf[choose][i]+ filter[5]*locbuf[choose][i+1] +filter[6]*locbuf[choose][i+width-1] + filter[7]*locbuf[choose][i+width]+ filter[8]*locbuf[choose][i+width+1] )/SUM;
	}
}

void apply_outer_filter(unsigned char** locbuf, unsigned int width, unsigned int height, unsigned char choose ,unsigned char* up,unsigned char*down,unsigned char* left,unsigned char* right,unsigned char ul,unsigned char ur,unsigned char dl, unsigned char dr){
	unsigned int i;
	unsigned char me;
//applies filter to pixels at the up,down,left,right margin of the image an calculates the upper right, upper left, down right, down left pixel
	me=locbuf[choose][0];
	if (up != NULL && left !=NULL)
		locbuf[1-choose][0]= calculate_value(ul, up[0], up[1], left[0], me, locbuf[choose][1] ,left[1],locbuf[choose][width],locbuf[choose][width+1]) ;
	else if (up == NULL && left !=NULL)
		locbuf[1-choose][0]= calculate_value(me,me,me, left[0], me, locbuf[choose][1] ,left[1],locbuf[choose][width],locbuf[choose][width+1]) ;
	else if (up != NULL && left ==NULL)
		locbuf[1-choose][0]= calculate_value(me, up[0], up[1], me, me, locbuf[choose][1] ,me,locbuf[choose][width],locbuf[choose][width+1]) ;
	else
		locbuf[1-choose][0]= calculate_value(me, me, me, me, me, locbuf[choose][1] ,me,locbuf[choose][width],locbuf[choose][width+1]) ;

	me=locbuf[choose][width*(height-1)];
	if (down != NULL && left !=NULL)
		locbuf[1-choose][width*(height-1)]= calculate_value(left[height-2], locbuf[choose][width*(height-2)], locbuf[choose][width*(height-2)+1], left[height-1], me, locbuf[choose][width*(height-1)+1], dl, down[0],down[1]);
	else if (down == NULL && left !=NULL)
		locbuf[1-choose][width*(height-1)]= calculate_value(left[height-2], locbuf[choose][width*(height-2)], locbuf[choose][width*(height-2)+1], left[height-1], me, locbuf[choose][width*(height-1)+1], me,me,me);
	else if (down != NULL && left ==NULL)
		locbuf[1-choose][width*(height-1)]= calculate_value(me, locbuf[choose][width*(height-2)], locbuf[choose][width*(height-2)+1], me, me, locbuf[choose][width*(height-1)+1],me, down[0],down[1]);
	else
		locbuf[1-choose][width*(height-1)]= calculate_value(me, locbuf[choose][width*(height-2)], locbuf[choose][width*(height-2)+1], me, me, locbuf[choose][width*(height-1)+1],me, me,me);

	me=locbuf[choose][width-1];
	if (up != NULL && right !=NULL)
		locbuf[1-choose][width-1]= calculate_value(up[width-2],up[width-1],ur,locbuf[choose][width-2], me,right[0],locbuf[choose][width-2+width],locbuf[choose][width-1+width],right[1]);
	else if (up == NULL && right !=NULL)
		locbuf[1-choose][width-1]= calculate_value(me,me,me,locbuf[choose][width-2], me,right[0],locbuf[choose][width-2+width],locbuf[choose][width-1+width],right[1]);
	else if (up != NULL && right ==NULL)
		locbuf[1-choose][width-1]= calculate_value(up[width-2],up[width-1],me,locbuf[choose][width-2], me,me,locbuf[choose][width-2+width],locbuf[choose][width-1+width],me);
	else
		locbuf[1-choose][width-1]= calculate_value(me,me,me,locbuf[choose][width-2], me,me,locbuf[choose][width-2+width],locbuf[choose][width-1+width],me);

	me=locbuf[choose][width*height-1];
	if (down != NULL && right !=NULL)
		locbuf[1-choose][height*width-1]= calculate_value(locbuf[choose][width*(height-1)-2],locbuf[choose][width*(height-1)-1],right[height-2], locbuf[choose][width*height-2], me,right[height-1],down[width-2],down[width-1],dr );
	else if	(down == NULL && right !=NULL)
		locbuf[1-choose][height*width-1]= calculate_value(locbuf[choose][width*(height-1)-2],locbuf[choose][width*(height-1)-1],right[height-2], locbuf[choose][width*height-2], me,right[height-1],me,me,me );
	else if (down != NULL && right ==NULL)
		locbuf[1-choose][height*width-1]= calculate_value(locbuf[choose][width*(height-1)-2],locbuf[choose][width*(height-1)-1],me, locbuf[choose][width*height-2], me,me,down[width-2],down[width-1],me );
	else
		locbuf[1-choose][height*width-1]= calculate_value(locbuf[choose][width*(height-1)-2],locbuf[choose][width*(height-1)-1],me, locbuf[choose][width*height-2], me,me,me,me,me );

	if (up !=NULL)
		for(i=1; i<width-1;i++)
			locbuf[1-choose][i]= calculate_value(up[i-1], up[i],up[i+1], locbuf[choose][i-1], locbuf[choose][i], locbuf[choose][i+1] ,locbuf[choose][i-1+width],locbuf[choose][i+width],locbuf[choose][i+width+1]) ;
	else
		for(i=1; i<width-1;i++)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i],locbuf[choose][i],locbuf[choose][i], locbuf[choose][i-1], locbuf[choose][i], locbuf[choose][i+1] ,locbuf[choose][i-1+width],locbuf[choose][i+width],locbuf[choose][i+width+1]) ;

	if (down !=NULL)
		for(i=width*(height-1)+1; i<width*height-1;i++)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i-width-1], locbuf[choose][i-width], locbuf[choose][i+1-width], locbuf[choose][i-1], locbuf[choose][i], locbuf[choose][i+1] ,down[i-width*(height-1)-1],down[i-width*(height-1)],down[i-width*(height-1)+1]) ;
	else
		for(i=width*(height-1)+1; i<width*height-1;i++)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i-width-1], locbuf[choose][i-width], locbuf[choose][i+1-width], locbuf[choose][i-1], locbuf[choose][i], locbuf[choose][i+1] ,locbuf[choose][i],locbuf[choose][i],locbuf[choose][i]) ;

	if (left != NULL)
		for(i=width; i<(height-1)*width; i+=width)
			locbuf[1-choose][i]= calculate_value(left[i/width -1], locbuf[choose][i-width], locbuf[choose][i-width+1],left[i/width ], locbuf[choose][i], locbuf[choose][i+1],left[i/width +1], locbuf[choose][i+width], locbuf[choose][i+width+1]);
	else
		for(i=width; i<(height-1)*width; i+=width)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i], locbuf[choose][i-width], locbuf[choose][i-width+1],locbuf[choose][i], locbuf[choose][i], locbuf[choose][i+1],locbuf[choose][i], locbuf[choose][i+width], locbuf[choose][i+width+1]);

	if (right != NULL)
		for(i=2*width-1; i<(height-1)*width; i+=width)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i-1-width], locbuf[choose][i-width], right[i/width -1], locbuf[choose][i-1], locbuf[choose][i], right[i/width],locbuf[choose][i-1+width], locbuf[choose][i+width], right[i/width +1]);
	else
		for(i=2*width-1; i<(height-1)*width; i+=width)
			locbuf[1-choose][i]= calculate_value(locbuf[choose][i-1-width], locbuf[choose][i-width], locbuf[choose][i], locbuf[choose][i-1], locbuf[choose][i], locbuf[choose][i],locbuf[choose][i-1+width], locbuf[choose][i+width], locbuf[choose][i]);
}

unsigned char** openRGBImage(char *path, unsigned int height, unsigned int width){
	unsigned char *temp = (unsigned char*) malloc(3 * height * width * sizeof(unsigned char));

	FILE* image_file = fopen(path, "r");
	if (image_file == NULL) {
		perror("fopen failed");
		return NULL;
	}

	if (fread(temp, sizeof(unsigned char), 3 * height * width, image_file) !=3* height * width) {
		fprintf(stderr, "fread failed");
		return NULL;
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
	free(image_array[0]);
	free(image_array[1]);
	free(image_array[2]);
	free(image_array);
	free(temp);
	return;
}

unsigned char* getGreyImageArray(char *path, unsigned int height, unsigned int width) {

	unsigned char *image_array = malloc(
			(height * width) * sizeof(unsigned char));

	FILE* image_file = fopen(path, "r");
	if (image_file == NULL) {
		perror("fopen failed");
		return NULL;
	}

	if ( fread(image_array, sizeof(unsigned char), height * width, image_file) !=height * width){
		perror("writting failed");
		return NULL;
	}
	fclose(image_file);

	return image_array;
}

void passGreyChangestoFile(char* path, void* image_array, unsigned int height,
		unsigned int width) {

	FILE* image_file = fopen(path, "w");
	if (image_file == NULL) {
		perror("fopen failed");
		exit(EXIT_FAILURE);
	}

	if (fwrite(image_array, sizeof(unsigned char), height *width, image_file) !=height *width){
		perror("write failed");
		exit(EXIT_FAILURE);
	}
	fclose(image_file);
	free(image_array);
	return;
}
