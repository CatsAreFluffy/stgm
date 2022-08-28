#include <iostream>
#include <cstdint>
#include <emmintrin.h>

//string to parent offset matrix
//returns number of columns
int parsematrix(const char* x, int8_t* y, int8_t* z) {
	// parse
	int8_t* entry=y;
	int8_t* column=y;
	int columns=0;
	for(;*x;x++){
		if(*x==','){
			entry++;
		}else if(*x==')'){
			column+=16;
			columns++;
			entry=column;
		}else if(*x!='('){
			*entry=(*entry)*10+(*x)-'0';
		}
	}
	// translate
	// first row
	for(int i=columns-1;i>=0;i--){
		int8_t entry=y[i*16];
		if(!entry){
			continue;
		}
		int8_t k=1;
		while(y[(i-k)*16]>=entry){
			k++;
		}
		y[i*16]=k;
		if (z) z[i]=0;
		// std::cout << i << "0 " << (int)k << std::endl << std::flush;
	}
	// others
	for(int j=1;j<16;j++){
		for(int i=columns-1;i>=0;i--){
			int8_t entry=y[i*16+j];
			if(!entry){
				continue;
			}
			int8_t k=y[i*16+j-1];
			while(y[(i-k)*16+j]>=entry){
				k+=y[(i-k)*16+j-1];
			}
			y[i*16+j]=k;
			if (z) z[i]=j;
			// std::cout << i << j << ' ' << (int)k << std::endl << std::flush;
		}
	}
	return columns;
}

void printmatrix(int8_t* x) {
	int8_t decode[16*10]={};
	for(int i=0;i<10;i++){
		std::cout << '(';
		for(int j=0;j<16;j++){
			std::cout << ((int)x[i*16+j]) << ',';
			if(x[i*16+j]>0){
				decode[i*16+j]=decode[(i+x[i*16+j])*16+j]+1;
			}
		}
		std::cout << ") (";
		for(int j=0;j<16;j++){
			std::cout << ((int)decode[i*16+j]) << ',';
		}
		std::cout << ')' << std::endl;
	}
	// std::cout << std::endl;
}

void printvcolumn(__m128i x) {
	__attribute__((aligned(16))) int8_t decode[16];
	_mm_store_si128((__m128i*)decode, x);
	std::cout << '(';
	for(int j=0;j<16;j++){
		std::cout << ((int)decode[j]) << ',';
	}
	std::cout << ')' << std::endl;
}

int main(int argc, const char* argv[]) {
	const int maxcol=10;
	const int maxlen=16*maxcol;
	// invariant: all entries past the end of the matrix are 0
	__attribute__((aligned(16))) int8_t matrix[maxlen]={};
	int8_t lnzs[maxcol]={};
	int mcols=parsematrix(argv[1], matrix, lnzs);
	// active column
	int8_t* mtail=matrix+mcols*16-16;
	int mtailx=mcols-1;
	__attribute__((aligned(16))) int8_t finish[maxlen]={};
	int fcols=parsematrix(argv[2], finish, NULL);
	int8_t* ftail=finish+fcols*16-16;
	// incremental compare pointers
	int8_t* mcmp=matrix;
	int8_t* fcmp=finish;
	// if(argc>3){
	// 	printmatrix(matrix);
	// 	std::cout << (mtail-matrix)/16 << std::endl;
	// }
	int_fast64_t steps=0;
	int_fast64_t zeroes=0;
	while(1){
		// advance compare pointers
		while(*mcmp==*fcmp){
			mcmp++;
			fcmp++;
			if(fcmp==finish+maxlen){
				goto bigbreak;
			}
		}
		// step until compare column updates
		int8_t* mupd=mcmp+1;
		while(mupd>mcmp){
			mupd=mtail;
			steps++;
			// if(steps>1){
			// 	matrix[45678]+=3;
			// }
			int8_t badroot=mtail[lnzs[mtailx]];
			// std::cout << (int)badroot << std::endl;
			if(!badroot){
				zeroes++;
				mtail-=16;
				mtailx--;
				// if(argc>3){
				// 	printmatrix(matrix);
				// 	std::cout << (mtail-matrix)/16 << ' ' << mcmp-matrix << std::endl;
				// }
				continue;
			}
			// copy the rest of the columns
			__m128i* mtailv=(__m128i*)mtail;
			// last column to copy to
			__m128i* mfinal=mtailv-1;
			int mfinalx=mtailx-1;
			while(mfinal+badroot<(__m128i*)(matrix+maxlen)){
				mfinal+=badroot;
				mfinalx+=badroot;
			}
			// std::cout << ((int8_t*)mfinal)-matrix << std::endl;
			// if no copies are made, clear the cut node
			if(mfinal==mtailv-1){
				for(int j=0;j<16;j++){
					mtail[j]=0;
				}
				// lnzs[mtailx]=0;
			}else{
				// the copy to the cut node needs to be handled separately,
				// since the pre-LNZ part isn't copied from the bad root and
				// the other part is never ascended
				// example: *mtail=ab00, *mtail+copyoffset=xyz0,
				// badroot=V=X-x=Y-y..., -1=F
				// 0000
				__m128i zero=_mm_setzero_si128();
				// ab00
				__m128i cutnode=*mtailv;
				// xyz0
				__m128i badrootcol=mtailv[-badroot];
				// FF00
				__m128i cutnodenz=_mm_cmpgt_epi8(cutnode,zero);
				// F000
				__m128i cutnodekeepmask=_mm_bsrli_si128(cutnodenz,1);
				// a000
				__m128i cutnodekeep=_mm_and_si128(cutnode,cutnodekeepmask);
				// 0yz0
				__m128i badrootlcol=_mm_andnot_si128(cutnodekeepmask,badrootcol);
				// 0FF0
				__m128i badrootnz=_mm_cmpgt_epi8(badrootlcol,zero);
				// VYZV
				__m128i badrootdescend=_mm_add_epi8(badrootlcol,_mm_set1_epi8(badroot));
				// 0YZ0
				__m128i badrootfinal=_mm_and_si128(badrootdescend,badrootnz);
				// aYZ0
				*mtailv=_mm_add_epi8(cutnodekeep,badrootfinal);
				// std::cout << "hi" << std::endl;
				lnzs[mtailx]=lnzs[mtailx]-1>lnzs[mtailx-badroot]?lnzs[mtailx]-1:lnzs[mtailx-badroot];
				// copy each column of the bad part (but use cut node instead
				// of bad root since the bad root was copied above)
				__m128i* badrootp=mtailv-badroot;
				int badrootpx=mtailx-badroot;
				// std::cout << "hi" << std::endl;
				for(int copycoli=badroot;copycoli>0;copycoli--){
					__m128i copycol=badrootp[copycoli];
					int copycolx=badrootpx+copycoli;
					__m128i descendm=_mm_cmpgt_epi8(copycol,_mm_set1_epi8(copycoli));
					__m128i descender=_mm_and_si128(descendm,_mm_set1_epi8(badroot));
					__m128i* target=mtailv+copycoli;
					int targetx=mtailx+copycoli;
					for(;target<=mfinal;target+=badroot,targetx+=badroot){
						copycol=_mm_add_epi8(copycol,descender);
						*target=copycol;
						lnzs[targetx]=lnzs[copycolx];
					}
				}
				// std::cout << "hi" << std::endl;
			}
			mtail=(int8_t*)mfinal;
			mtailx=mfinalx;
			// if(argc>3){
			// 	printmatrix(matrix);
			// 	std::cout << (mtail-matrix)/16 << ' ' << mcmp-matrix << std::endl;
			// }
		}
	}
	bigbreak:
	// there may be 0 columns in the matrix that need to be accounted for
	int extracols=((mtail-matrix)-(ftail-finish))/16;
	steps+=extracols;
	zeroes+=extracols;
	std::cout << "Steps: " << steps << std::endl;
	std::cout << "Counter: " << zeroes << std::endl;
	return 0;
}
