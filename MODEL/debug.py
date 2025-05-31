import numpy as np
import os
from joblib import delayed, Parallel, cpu_count
from tqdm import tqdm
import MODEL.modeling_utils as modutl

def upsample_fixed(sig: np.ndarray,
                  MODELts: np.ndarray,
                  MODELsize: int,
                  SIGsize: int,
                  ITERneuron: np.ndarray,
                  msdelays: np.ndarray,
                  gaussSD: float = 20,
                  additionalMSoffset: float = 5,
                  ) -> tuple[np.ndarray, np.ndarray]:
    
    # Debug info
    print(f"Input shapes: sig={sig.shape}, MODELts={MODELts.shape}")
    print(f"MODELsize={MODELsize}, SIGsize={SIGsize}")
    print(f"ITERneuron shape={ITERneuron.shape}, msdelays shape={msdelays.shape}")
    
    # Account for msdelays based on location within imaging frame
    bindelays = np.round(msdelays)
    SPACING = MODELsize // SIGsize
    
    # Try smaller batch sizes to reduce memory overhead
    n_workers = min(cpu_count(), 8)  # Limit workers
    batches = np.array_split(ITERneuron, n_workers)
    
    print(f"Using {n_workers} workers, batch sizes: {[b.size for b in batches]}")
    
    worker_args = [
        [sig[:, batch], MODELts,
         batch.size,
         SPACING, MODELsize,
         gaussSD,
         bindelays[batch], additionalMSoffset
        ]
        for batch in batches
    ]
    
    # FIXED: Use the correct worker function
    print("Starting parallel processing...")
    outList = Parallel(n_jobs=n_workers, verbose=14, backend='loky')(
        delayed(modutl.upsampling_worker)(*wa) for wa in worker_args
    )
    
    return np.column_stack(outList)

# Alternative version with memory optimization
def upsample_memory_optimized(sig: np.ndarray,
                             MODELts: np.ndarray,
                             MODELsize: int,
                             SIGsize: int,
                             ITERneuron: np.ndarray,
                             msdelays: np.ndarray,
                             gaussSD: float = 20,
                             additionalMSoffset: float = 5,
                             ) -> tuple[np.ndarray, np.ndarray]:
    
    # Force single-threaded NumPy to avoid conflicts
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    bindelays = np.round(msdelays)
    SPACING = MODELsize // SIGsize
    
    # Use fewer workers for memory-intensive tasks
    n_workers = min(cpu_count() // 2, 8)
    batches = np.array_split(ITERneuron, n_workers)
    
    print(f"Memory-optimized version using {n_workers} workers")
    
    worker_args = [
        [sig[:, batch].copy(),  # Make explicit copies to avoid sharing issues
         MODELts.copy(),
         batch.size,
         SPACING, MODELsize,
         gaussSD,
         bindelays[batch].copy(), additionalMSoffset
        ]
        for batch in batches
    ]
    
    # Try different backends
    for backend in ['loky', 'threading', 'multiprocessing']:
        try:
            print(f"Trying backend: {backend}")
            outList = Parallel(n_jobs=n_workers, verbose=10, backend=backend)(
                delayed(modutl.upsampling_worker)(*wa) for wa in worker_args
            )
            print(f"Success with {backend} backend!")
            break
        except Exception as e:
            print(f"Failed with {backend}: {e}")
            continue
    else:
        print("All backends failed, falling back to sequential processing")
        outList = [modutl.upsampling_worker(*wa) for wa in worker_args]
    
    return np.column_stack(outList)

# Debugging version to test step by step
def debug_upsampling_issue(sig: np.ndarray,
                          MODELts: np.ndarray,
                          MODELsize: int,
                          SIGsize: int,
                          ITERneuron: np.ndarray,
                          msdelays: np.ndarray,
                          gaussSD: float = 20,
                          additionalMSoffset: float = 5):
    
    print("=== DEBUGGING UPSAMPLING ISSUE ===")
    
    # Check data sizes
    sig_size_mb = sig.nbytes / (1024**2)
    modelts_size_mb = MODELts.nbytes / (1024**2)
    print(f"Data sizes: sig={sig_size_mb:.1f}MB, MODELts={modelts_size_mb:.1f}MB")
    
    if sig_size_mb > 100:
        print("WARNING: Large signal array detected, this may cause memory issues")
    
    # Test with minimal data first
    print("\n1. Testing with single neuron...")
    test_batch = ITERneuron[:1]
    test_args = [sig[:, test_batch], MODELts, 1, MODELsize // SIGsize, 
                MODELsize, gaussSD, np.round(msdelays[test_batch]), additionalMSoffset]
    
    try:
        result = modutl.upsampling_worker(*test_args)
        print(f"Single neuron test SUCCESS: output shape {result.shape}")
    except Exception as e:
        print(f"Single neuron test FAILED: {e}")
        return None
    
    # Test with small parallel batch
    print("\n2. Testing with 2 workers, small batch...")
    small_batch = ITERneuron[:4] if len(ITERneuron) > 4 else ITERneuron
    batches = np.array_split(small_batch, 2)
    worker_args = [
        [sig[:, batch], MODELts, batch.size, MODELsize // SIGsize,
         MODELsize, gaussSD, np.round(msdelays[batch]), additionalMSoffset]
        for batch in batches
    ]
    
    try:
        outList = Parallel(n_jobs=2, verbose=10, backend='loky')(
            delayed(modutl.upsampling_worker)(*wa) for wa in worker_args
        )
        print("Small parallel test SUCCESS!")
        return np.column_stack(outList)
    except Exception as e:
        print(f"Small parallel test FAILED: {e}")
        
        # Try sequential as fallback
        print("Trying sequential processing...")
        outList = [modutl.upsampling_worker(*wa) for wa in worker_args]
        print("Sequential processing SUCCESS!")
        return np.column_stack(outList)