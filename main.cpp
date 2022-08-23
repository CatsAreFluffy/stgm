#include <iostream>
#include <cstdint>
#include <emmintrin.h>

//string to parent offset matrix
//returns number of columns
int parsematrix(const char* x, int8_t* y) {
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
		int8_t k=-1;
		while(y[(i+k)*16]>=entry){
			k--;
		}
		y[i*16]=k;
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
			while(y[(i+k)*16+j]>=entry){
				k+=y[(i+k)*16+j-1];
			}
			y[i*16+j]=k;
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
			std::cout << ((int)-x[i*16+j]) << ',';
			if(x[i*16+j]<0){
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
	const int maxlen=16*10;
	// invariant: all entries past the end of the matrix are 0
	__attribute__((aligned(16))) int8_t matrix[maxlen]={};
	int mcols=parsematrix(argv[1], matrix);
	// active column
	int8_t* mtail=matrix+mcols*16-16;
	__attribute__((aligned(16))) int8_t finish[maxlen]={};
	int fcols=parsematrix(argv[2], finish);
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
			// if(steps>2){
			// 	matrix[45678]+=3;
			// }
			int8_t badroot=0;
			// used later
			int i=0;
			// fails for 16 row columns
			for(;mtail[i];i++){
				badroot=mtail[i];
			}
			if(!badroot){
				zeroes++;
				mtail-=16;
				// if(argc>3){
				// 	printmatrix(matrix);
				// 	std::cout << (mtail-matrix)/16 << ' ' << mcmp-matrix << std::endl;
				// }
				continue;
			}
			int copyoffset=badroot*16;
			// the copy to the cut node needs to be handled separately,
			// since the pre-LNZ part isn't copied from the bad root and
			// the other part is never ascended
			// equivalent scalar code:
			// for(i--;i<16;i++){
			// 	mtail[i]=mtail[i+copyoffset]+(mtail[i+copyoffset]?badroot:0);
			// }
			// example: *mtail=ab00, *mtail+copyoffset=xyz0,
			// badroot=V=X-x=Y-y..., -1=F
			// 0000
			__m128i zero=_mm_setzero_si128();
			// ab00
			__m128i cutnode=*(__m128i*)mtail;
			// xyz0
			__m128i badrootcol=*(__m128i*)(mtail+copyoffset);
			// FF00
			__m128i cutnodenz=_mm_cmplt_epi8(cutnode,zero);
			// F000
			__m128i cutnodekeepmask=_mm_bsrli_si128(cutnodenz,1);
			// a000
			__m128i cutnodekeep=_mm_and_si128(cutnode,cutnodekeepmask);
			// 0yz0
			__m128i badrootlcol=_mm_andnot_si128(cutnodekeepmask,badrootcol);
			// 0FF0
			__m128i badrootnz=_mm_cmplt_epi8(badrootlcol,zero);
			// VYZV
			__m128i badrootdescend=_mm_add_epi8(badrootlcol,_mm_set1_epi8(badroot));
			// 0YZ0
			__m128i badrootfinal=_mm_and_si128(badrootdescend,badrootnz);
			// aYZ0
			__m128i newcol=_mm_add_epi8(cutnodekeep,badrootfinal);
			*(__m128i*)mtail=newcol;
			i=16;
			int descendlim=0;
			while(mtail-copyoffset<=matrix+maxlen){
				// the copy to the cut node has already been handled
				// so skip that here
				for(;i<-copyoffset;i++){
					int8_t entry=mtail[i+copyoffset];
					if(entry<descendlim-i/16){
						entry+=badroot;
					}
					mtail[i]=entry;
				}
				mtail-=copyoffset;
				descendlim+=badroot;
				i=0;
			}
			// if no copies are made, clear the cut node
			if(i>0){
				for(int j=0;j<16;j++){
					mtail[j]=0;
				}
			}
			mtail-=16;
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