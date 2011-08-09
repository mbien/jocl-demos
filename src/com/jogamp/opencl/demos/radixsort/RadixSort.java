/*
 * 20:38 Sunday, February 28 2010
 */

package com.jogamp.opencl.demos.radixsort;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLResource;
import java.io.IOException;
import java.nio.IntBuffer;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static com.jogamp.opencl.CLProgram.*;
import static com.jogamp.opencl.CLProgram.CompilerOptions.*;

/**
 *
 * @author Michael Bien
 */
public class RadixSort implements CLResource {

    private static final int NUM_BANKS = 16;
    private static final int WARP_SIZE = 32;
    private static final int bitStep   = 4;

    private final int CTA_SIZE;

    private final CLKernel ckRadixSortBlocksKeysOnly;
    private final CLKernel ckFindRadixOffsets;
    private final CLKernel ckScanNaive;
    private final CLKernel ckReorderDataKeysOnly;

    private final CLBuffer<?> tempKeys;
    private final CLBuffer<?> mCounters;
    private final CLBuffer<?> mCountersSum;
    private final CLBuffer<?> mBlockOffsets;

    private final CLCommandQueue queue;
    private final Scan scan;
    private final CLProgram program;

    public RadixSort(CLCommandQueue queue, int maxElements, int CTA_SIZE) throws IOException {

        this.CTA_SIZE = CTA_SIZE;
        scan = new Scan(queue, maxElements / 2 / CTA_SIZE * 16);

        int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0)
                ? (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);

        this.queue = queue;

        CLContext context  = queue.getContext();
        this.tempKeys      = context.createBuffer(4 * maxElements,           READ_WRITE);
        this.mCounters     = context.createBuffer(4 * WARP_SIZE * numBlocks, READ_WRITE);
        this.mCountersSum  = context.createBuffer(4 * WARP_SIZE * numBlocks, READ_WRITE);
        this.mBlockOffsets = context.createBuffer(4 * WARP_SIZE * numBlocks, READ_WRITE);

        program = context.createProgram(getClass().getResourceAsStream("RadixSort.cl"))
                         .build(ENABLE_MAD, define("WARP_SIZE", WARP_SIZE));

//        out.println(program.getBuildLog());

        ckRadixSortBlocksKeysOnly  = program.createCLKernel("radixSortBlocksKeysOnly");
        ckFindRadixOffsets         = program.createCLKernel("findRadixOffsets");
        ckScanNaive                = program.createCLKernel("scanNaive");
        ckReorderDataKeysOnly      = program.createCLKernel("reorderDataKeysOnly");

    }

    void sort(CLBuffer<IntBuffer> d_keys, int numElements, int keyBits) {
        radixSortKeysOnly(d_keys, numElements, keyBits);
    }

    //----------------------------------------------------------------------------
    // Main key-only radix sort function.  Sorts in place in the keys and values
    // arrays, but uses the other device arrays as temporary storage.  All pointer
    // parameters are device pointers.  Uses cudppScan() for the prefix sum of
    // radix counters.
    //----------------------------------------------------------------------------
    void radixSortKeysOnly(CLBuffer<IntBuffer> keys, int numElements, int keyBits) {
        int i = 0;
        while (keyBits > i * bitStep) {
            radixSortStepKeysOnly(keys, bitStep, i * bitStep, numElements);
            i++;
        }
    }

    //----------------------------------------------------------------------------
    // Perform one step of the radix sort.  Sorts by nbits key bits per step,
    // starting at startbit.
    //----------------------------------------------------------------------------
    void radixSortStepKeysOnly(CLBuffer<IntBuffer> keys, int nbits, int startbit, int numElements) {

        // Four step algorithms from Satish, Harris & Garland
        radixSortBlocksKeysOnlyOCL(keys, nbits, startbit, numElements);

        findRadixOffsetsOCL(startbit, numElements);

        scan.scanExclusiveLarge(mCountersSum, mCounters, 1, numElements / 2 / CTA_SIZE * 16);

        reorderDataKeysOnlyOCL(keys, startbit, numElements);
    }

    //----------------------------------------------------------------------------
    // Wrapper for the kernels of the four steps
    //----------------------------------------------------------------------------
    void radixSortBlocksKeysOnlyOCL(CLBuffer<IntBuffer> keys, int nbits, int startbit, int numElements) {

        int totalBlocks = numElements / 4 / CTA_SIZE;
        int globalWorkSize = CTA_SIZE * totalBlocks;
        int localWorkSize = CTA_SIZE;

        ckRadixSortBlocksKeysOnly.putArg(keys).putArg(tempKeys).putArg(nbits).putArg(startbit)
                                 .putArg(numElements).putArg(totalBlocks).putArgSize(4 * CTA_SIZE * 4)
                                 .rewind();

        queue.put1DRangeKernel(ckRadixSortBlocksKeysOnly, 0, globalWorkSize, localWorkSize);
    }

    void findRadixOffsetsOCL(int startbit, int numElements) {

        int totalBlocks = numElements / 2 / CTA_SIZE;
        int globalWorkSize = CTA_SIZE * totalBlocks;
        int localWorkSize = CTA_SIZE;

        ckFindRadixOffsets.putArg(tempKeys).putArg(mCounters).putArg(mBlockOffsets)
                          .putArg(startbit).putArg(numElements).putArg(totalBlocks).putArgSize(2 * CTA_SIZE * 4)
                          .rewind();

        queue.put1DRangeKernel(ckFindRadixOffsets, 0, globalWorkSize, localWorkSize);
    }

    void scanNaiveOCL(int numElements) {
        
        int nHist = numElements / 2 / CTA_SIZE * 16;
        int globalWorkSize = nHist;
        int localWorkSize = nHist;
        int extra_space = nHist / NUM_BANKS;
        int shared_mem_size = 4 * (nHist + extra_space);

        ckScanNaive.putArg(mCountersSum).putArg(mCounters).putArg(nHist).putArgSize(2 * shared_mem_size).rewind();

        queue.put1DRangeKernel(ckScanNaive, 0, globalWorkSize, localWorkSize);
    }

    void reorderDataKeysOnlyOCL(CLBuffer<IntBuffer> keys, int startbit, int numElements) {

        int totalBlocks = numElements / 2 / CTA_SIZE;
        int globalWorkSize = CTA_SIZE * totalBlocks;
        int localWorkSize = CTA_SIZE;

        ckReorderDataKeysOnly.putArg(keys).putArg(tempKeys).putArg(mBlockOffsets).putArg(mCountersSum).putArg(mCounters)
                             .putArg(startbit).putArg(numElements).putArg(totalBlocks).putArgSize(2 * CTA_SIZE * 4).rewind();

        queue.put1DRangeKernel(ckReorderDataKeysOnly, 0, globalWorkSize, localWorkSize);
    }

    public void release() {

        scan.release();

        //program & kernels
        program.release();

        //buffers
        tempKeys.release();
        mCounters.release();
        mCountersSum.release();
        mBlockOffsets.release();
    }

    @Override
    public boolean isReleased() {
        return scan.isReleased();
    }

    public void close() {
        release();
    }



}
