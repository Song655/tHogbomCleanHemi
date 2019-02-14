// Include own header file first
#include "HogbomHemi.h"

// System includes
#include <vector>
#include <iostream>
#include <cstddef>
#include <cmath>

// Local includes
#include "Parameters.h"

using namespace std;


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


template <unsigned int blockSize>
__global__ void findPeak(float *g_idata, float *g_oValue, unsigned int *g_oIndex, unsigned int size) {

	    __shared__ float s_value[blockSize];
	    __shared__ unsigned int s_index[blockSize];

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x* blockSize*2 + tid;
		unsigned int gridSize = blockSize*2*gridDim.x;
		s_value[tid] = 0;
		s_index[tid] = 0;
		float d1, d2;

		while ((i+blockSize) < size) {
			d1 = fabsf(g_idata[i]);
			d2 = fabsf(g_idata[i+blockSize]);
			if(d2 > d1){
				s_value[tid]= g_idata[i+blockSize];
				s_index[tid] = i+blockSize;
			}
			else{
				s_value[tid] = g_idata[i];
				s_index[tid] = i;
			}
			i += gridSize;
		}__syncthreads();

		// do reduction in shared mem
		if (blockSize >= 512) {
		    if (tid < 256) {
				d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+256]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+256];
					s_index[tid] = s_index[tid+256];
				}
		    } __syncthreads();
		}
		if (blockSize >= 256) {
		    if (tid < 128) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+128]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+128];
					s_index[tid] = s_index[tid+128];
				}
		    } __syncthreads();
		}
		if (blockSize >= 128) {
		    if (tid < 64) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+64]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+64];
					s_index[tid] = s_index[tid+64];
				}
		    } __syncthreads();
		}
		if (tid < 32) {
		    if (blockSize >= 64) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+32]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+32];
					s_index[tid] = s_index[tid+32];
				}
		    }
		    if (blockSize >= 32) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+16]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+16];
					s_index[tid] = s_index[tid+16];
				}
		    }
		    if (blockSize >= 16){
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+8]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+8];
					s_index[tid] = s_index[tid+8];
				}
		    }
		    if (blockSize >= 8){
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+4]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+4];
					s_index[tid] = s_index[tid+4];
				}
		    }
		    if (blockSize >= 4) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+2]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+2];
					s_index[tid] = s_index[tid+2];
				}
		    }
		    if (blockSize >= 2) {
		    	d1 = fabsf(s_value[tid]);
				d2 = fabsf(s_value[tid+1]);
				if(d2 > d1) {
					s_value[tid] = s_value[tid+1];
					s_index[tid] = s_index[tid+1];
				}
		    }
		}
		// write result for this block to global mem
		if (tid == 0) {
			g_oValue[blockIdx.x] = s_value[0];
			g_oIndex[blockIdx.x] = s_index[0];
		}
}

void gpufindPeak(float *data, unsigned size,float& maxVal, int& maxPos)
{

	float *d_idata;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_idata, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_idata, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	const int THREAD_NUM = 512;
	const int BLOCK_Count  =ceil(1.0*size/THREAD_NUM);

	float* d_oValue,*result_val;
	unsigned int *d_oIndex, *result_index;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_oValue, sizeof(float)*BLOCK_Count));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_oIndex, sizeof(unsigned int)*BLOCK_Count));

	result_val = (float*)malloc(sizeof(float)*BLOCK_Count);
	result_index =(unsigned int*) malloc(sizeof(unsigned int)*BLOCK_Count);

	switch (THREAD_NUM)
	{
		case 512:
			findPeak <512><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 256:
			findPeak <256><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 128:
			findPeak <128><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 64:
			findPeak <64><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 32:
			findPeak <32><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 16:
			findPeak <16><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 8:
			findPeak <8><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 4:
			findPeak <4><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex,size); break;
		case 2:
			findPeak <2><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex, size); break;
		case 1:
			findPeak <1><<<BLOCK_Count, THREAD_NUM>>>(d_idata, d_oValue,d_oIndex,size); break;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(result_val, d_oValue, sizeof(float)*BLOCK_Count, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(result_index, d_oIndex, sizeof(unsigned int)*BLOCK_Count, cudaMemcpyDeviceToHost));


	CUDA_CHECK_RETURN(cudaFree(d_idata));
	CUDA_CHECK_RETURN(cudaFree(d_oValue));
	CUDA_CHECK_RETURN(cudaFree(d_oIndex));

	maxVal = 0.0;
	maxPos = 0;
	for (int i = 0; i < BLOCK_Count; i++) {
		if (abs(result_val[i]) > abs(maxVal)) {
			maxVal = result_val[i];
			maxPos = result_index[i];
		}
	}
	
	free(result_val);
	free(result_index);

}

__global__ static void deconvolve(const vector<float>& dirty,
                            const size_t dirtyWidth,    
                            const size_t psfWidth,
                            vector<float>& model,
                            vector<float>& residual)
{		
		extern __shared__ int shared[];	
		const int tid = threadIdx.x;	
		const int bid = blockIdx.x;	

		// Find the peak of the PSF

	
		int i;
		if(tid == 0) time[bid] = clock();	
		shared[tid] = 0;	
		for( i = bid*THREAD_NUM+tid; i<DATA_SIZE; i += BLOCK_NUM*THREAD_NUM){		
		shared[tid] += num[i]*num[i];	
		}	

		__syncthreads();	
		if(tid == 0){		
		for(i = 1; i < THREAD_NUM; i++){		
		shared[0] += shared[i];		}	
		result[bid] = shared[0];	
		}		
		if(tid ==0 ) time[bid+BLOCK_NUM] = clock();
}

void HogbomHemi::deconvolve(const vector<float>& dirty,
                            const size_t dirtyWidth,    
                            const size_t psfWidth,
                            vector<float>& model,
                            vector<float>& residual)
{
    residual = dirty;
    hemi::Array<float> d_residual(&residual[0], residual.size());

    // Find the peak of the PSF
    MaxCandidate psfPeak = {0.0, 0};
    findPeak(m_psf->readOnlyPtr(), m_psf->size(), psfPeak);
    
    cout << "Found peak of PSF: " << "Maximum = " << psfPeak.value
         << " at location " << idxToPos(psfPeak.index, psfWidth).x << ","
         << idxToPos(psfPeak.index, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        MaxCandidate absPeak = {0.0, 0};
        findPeak(d_residual.readOnlyPtr(), d_residual.size(), absPeak);

        //cout << "Iteration: " << i + 1 << " - Maximum = " << absPeak.value
        //    << " at location " << idxToPos(absPeak.index, dirtyWidth).x << ","
        //    << idxToPos(absPeak.index, dirtyWidth).y << endl;

        // Check if threshold has been reached
        if (abs(absPeak.value) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Add to model
        model[absPeak.index] += absPeak.value * g_gain;

        // Subtract the PSF from the residual image
        subtractPSF(m_psf->readOnlyPtr(), psfWidth, d_residual.ptr(), dirtyWidth, 
                    absPeak.index, psfPeak.index, absPeak.value, g_gain);
    }

    // force copy of residual back to host
    d_residual.readOnlyHostPtr();
}

HEMI_DEV_CALLABLE_INLINE
int blockId()
{
#ifdef HEMI_CUDA_COMPILER
    return blockIdx.x;
#else
    return omp_get_thread_num();
#endif
}

// For CUB
struct MaxOp
{
    HEMI_DEV_CALLABLE_INLINE_MEMBER
    MaxCandidate operator()(const MaxCandidate &a, const MaxCandidate &b)
    {
        return (abs(b.value) > abs(a.value)) ? b : a;
    }
};

HEMI_DEV_CALLABLE_INLINE
void findPeakReduce(MaxCandidate *peak, MaxCandidate threadMax)
{
#ifdef HEMI_DEV_CODE
    typedef cub::BlockReduce<MaxCandidate, HogbomHemi::FindPeakBlockSize> BlockMax;
    __shared__ typename BlockMax::TempStorage temp_storage;
    MaxOp op;
    threadMax = BlockMax(temp_storage).Reduce(threadMax, op);
    if (threadIdx.x == 0)
#endif
    {           
        peak[blockId()] = threadMax;
    }
}

HEMI_KERNEL(findPeakLoop)(MaxCandidate *peak, const float* image, int size)
{
    #pragma omp parallel
    {
        MaxCandidate threadMax = {0.0, 0};
        
        // parallel raking reduction (independent threads)
        #pragma omp for schedule(static)
        for (int i = hemiGetElementOffset(); i < size; i += hemiGetElementStride()) {
            if (abs(image[i]) > abs(threadMax.value)) {
                threadMax.value = image[i];
                threadMax.index = i;
            }
        }

        findPeakReduce(peak, threadMax);
    }
}

void HogbomHemi::findPeak(const float* image, size_t size, MaxCandidate &peak)
{
    HEMI_KERNEL_LAUNCH(findPeakLoop, m_findPeakNBlocks, FindPeakBlockSize, 0, 0,
                       m_blockMax->writeOnlyPtr(), image, size);   

    const MaxCandidate *maximum = m_blockMax->readOnlyHostPtr();
    
    peak = maximum[0];

    // serial final reduction
    for (int i = 1; i < m_findPeakNBlocks; ++i) {
        if (abs(maximum[i].value) > abs(peak.value))
            peak = maximum[i];
    }

}

HEMI_KERNEL(subtractPSFLoop)(const float* psf, const int psfWidth,
                             float* residual, const int residualWidth,
                             const int startx, const int starty,
                             int const stopx, const int stopy,
                             const int diffx, const int diffy,
                             const float absPeakVal,
                             const float gain)
{
    #pragma omp parallel for default(shared) schedule(static)
    for (int y = starty + hemiGetElementYOffset(); y <= stopy; y += hemiGetElementYStride()) {
        for (int x = startx + hemiGetElementXOffset(); x <= stopx; x += hemiGetElementXStride()) {
            residual[HogbomHemi::posToIdx(residualWidth, HogbomHemi::Position(x, y))] -= absPeakVal * gain
                * psf[HogbomHemi::posToIdx(psfWidth, HogbomHemi::Position(x - diffx, y - diffy))];
        }
    }    
}

void HogbomHemi::subtractPSF(const float* psf,
                             const size_t psfWidth,
                             float* residual,
                             const size_t residualWidth,
                             const size_t peakPos, 
                             const size_t psfPeakPos,
                             const float absPeakVal,
                             const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - py;

    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    const dim3 blockDim(32, 4);
        
    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx-startx+1.0f) / static_cast<float>(blockDim.x));
    const int blocksy = ceil((stopy-starty+1.0f) / static_cast<float>(blockDim.y));
    const dim3 gridDim(blocksx, blocksy);

    HEMI_KERNEL_LAUNCH(subtractPSFLoop, gridDim, blockDim, 0, 0, 
                       psf, psfWidth, residual, residualWidth, 
                       startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
}


void HogbomHemi::reportDevice(void)
{
    std::cout << "+++++ Forward processing (CUDA) +++++" << endl;
    // Report the type of device being used
    int device;
    cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << "( " << devprop.multiProcessorCount << " SMs)" << std::endl;

    m_findPeakNBlocks = 2 * devprop.multiProcessorCount;

}























