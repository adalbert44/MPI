#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <mpi.h>
#include <fstream>
#include <iostream>

using namespace std;

int ProcNum = 0; // Number of available processes
int ProcRank = 0; // Rank of current process

// Function for memory allocation and data initialization
void ProcessInitialization(string& ppmFormat,
	int& width, int& height, int& maxColor, unsigned char*&image, int& RowNum, unsigned char*& pProcRows, unsigned char*& pProcResult, unsigned char*& res) {
	
	if (ProcRank == 0) {
		ifstream input;
		input.open("C:\\Users\\38067\\source\\repos\\PictureBinarization\\inputPicture.ppm", ios::binary);
		if (!input.is_open()) {
			cout << "Can't open input file";
			exit(1);
		}
		input >> ppmFormat;
		input >> width >> height;
		input >> maxColor;

		image = new unsigned char[height * width];

		char r;
		input.get(r);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				char r, g, b;
				input.get(r);
				input.get(g);
				input.get(b);
				image[i * width + j] = 0.2125 * double(r) + 0.7154 * double(g) + 0.0721 * double(b);
			}
		}
		input.close();
	}

	int RestRows; // Number of rows, that haven’t been distributed yet
	int i; // Loop variable
	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

	RestRows = height;
	for (i = 0; i < ProcRank; i++)
		RestRows = RestRows - RestRows / (ProcNum - i);
	RowNum = RestRows / (ProcNum - ProcRank);
	pProcRows = new unsigned char[RowNum * width];
	pProcResult = new unsigned char[RowNum * width];
	res = new unsigned char[height * width];
}
// Data distribution among the processes
void DataDistribution(unsigned char* image, unsigned char* procRows, int width, int height, int RowNum) {
	int* pSendNum; // the number of elements sent to the process
	int* pSendInd; // the index of the first data element sent to the process
	int RestRows = height; // Number of rows, that haven’t been distributed yet
	// Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];
	// Define the disposition of the matrix rows for current process
	RowNum = (height / ProcNum);
	pSendNum[0] = RowNum * width;
	pSendInd[0] = 0;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pSendNum[i] = RowNum * width;
		pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
	}
	// Scatter the rows
	MPI_Scatterv(image, pSendNum, pSendInd, MPI_UNSIGNED_CHAR, procRows,
		pSendNum[ProcRank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
}
// Function for gathering the result vector
void ResultReplication(unsigned char* pProcResult, unsigned char* pResult, int width, int height,
	int RowNum) {
	int* pReceiveNum; // Number of elements, that current process sends
	int* pReceiveInd; /* Index of the first element from current process
	in result vector */
	int RestRows = height; // Number of rows, that haven’t been distributed yet
	//Alloc memory for temporary objects
	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];
	//Define the disposition of the result vector block of current processor
	RowNum = (height / ProcNum);
	pReceiveInd[0] = 0;
	pReceiveNum[0] = RowNum * width;
	for (int i = 1; i < ProcNum; i++) {
		RestRows -= RowNum;
		RowNum = RestRows / (ProcNum - i);
		pReceiveNum[i] = RowNum * width;
		pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
	}
	//Gather the whole result vector on every processor
	MPI_Allgatherv(pProcResult, pReceiveNum[ProcRank], MPI_UNSIGNED_CHAR, pResult,
		pReceiveNum, pReceiveInd, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
	//Free the memory
	delete[] pReceiveNum;
	delete[] pReceiveInd;
}
// Function for computational process termination
void ProcessTermination(string& ppmFormat,
	int& width, int& height, int& maxColor, unsigned char*& image, unsigned char*& res, unsigned char*& pProcRows, unsigned char*& pProcResult) {
	
	if (ProcRank == 0) {
		fstream output;

		output.open("C:\\Users\\38067\\source\\repos\\PictureBinarization\\outputPicture.ppm", ios::out | ios::binary);
		output << ppmFormat << '\n';
		output << width << ' ' << height << '\n';
		output << maxColor << '\n';
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				output << (char)res[i * width + j] << (char)res[i * width + j] << (char)res[i * width + j];
			}
		}

		output.close();
		delete[] image;
	}

	delete[] pProcRows;
	delete[] pProcResult;
	delete[] res;
}

void BradleyThreshold(unsigned char* src, unsigned char* res, int width, int height) {
	const int S = width / 8;
	int s2 = S / 2;
	const float t = 0.30;
	unsigned long* integral_image = 0;
	long sum = 0;
	int count = 0;
	int index;
	int x1, y1, x2, y2;

	integral_image = new unsigned long[width * height * sizeof(unsigned long*)];

	for (int i = 0; i < width; i++) {
		sum = 0;
		for (int j = 0; j < height; j++) {
			index = j * width + i;
			sum += src[index];
			if (i == 0)
				integral_image[index] = sum;
			else
				integral_image[index] = integral_image[index - 1] + sum;
		}
	}

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			index = j * width + i;

			x1 = i - s2;
			x2 = i + s2;
			y1 = j - s2;
			y2 = j + s2;

			if (x1 < 0)
				x1 = 0;
			if (x2 >= width)
				x2 = width - 1;
			if (y1 < 0)
				y1 = 0;
			if (y2 >= height)
				y2 = height - 1;

			count = (x2 - x1) * (y2 - y1);

			sum = integral_image[y2 * width + x2] - integral_image[y1 * width + x2] -
				integral_image[y2 * width + x1] + integral_image[y1 * width + x1];
			if ((long)(src[index] * count) < (long)(sum * (1.0 - t)))
				res[index] = 0;
			else
				res[index] = 255;
		}
	}

	delete[] integral_image;
}

void Serial() {
	string ppmFormat;
	int width, height;
	int maxColor;
	unsigned char* image;
	unsigned char* procImage;
	unsigned char* procRes;
	unsigned char* res;
	double Start, Finish;

	ifstream input;
	input.open("C:\\Users\\38067\\source\\repos\\PictureBinarization\\inputPicture.ppm", ios::binary);
	if (!input.is_open()) {
		cout << "Can't open input file";
		exit(1);
	}
	input >> ppmFormat;
	input >> width >> height;
	input >> maxColor;

	image = new unsigned char[height * width];

	char r;
	input.get(r);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			char r, g, b;
			input.get(r);
			input.get(g);
			input.get(b);
			image[i * width + j] = 0.2125 * double(r) + 0.7154 * double(g) + 0.0721 * double(b);
		}
	}
	input.close();

	res = new unsigned char[height * width];

	Start = clock();
	BradleyThreshold(image, res, width, height);
	Finish = clock();
	printf("Time of execution = %f\n", (Finish - Start) / CLK_TCK);

	fstream output;

	output.open("C:\\Users\\38067\\source\\repos\\PictureBinarization\\outputPicture.ppm", ios::out | ios::binary);
	output << ppmFormat << '\n';
	output << width << ' ' << height << '\n';
	output << maxColor << '\n';
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output << (char)res[i * width + j] << (char)res[i * width + j] << (char)res[i * width + j];
		}
	}

	output.close();
	delete[] image;
	delete[] res;
}

void main(int argc, char* argv[]) {
//	Serial();
	string ppmFormat;
	int width, height;
	int maxColor, rowNum;
	unsigned char* image;
	unsigned char* procImage;
	unsigned char* procRes;
	unsigned char* res;
	double Start, Finish;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	ProcessInitialization(ppmFormat, width, height, maxColor, image, rowNum, procImage, procRes, res);

	Start = clock();
	DataDistribution(image, procImage, width, height, rowNum);
	BradleyThreshold(procImage, procRes, width, rowNum);
	ResultReplication(procRes, res, width, height, rowNum);

	Finish = clock();
	ProcessTermination(ppmFormat, width, height, maxColor, image, res, procImage, procRes);

	if (ProcRank == 0) {
		printf("Time of execution = %f\n", (Finish - Start) / CLK_TCK);
	}

	MPI_Finalize();
	
}