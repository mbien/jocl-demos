/*
 * 18:42 Saturday, February 27 2010
 */
package com.jogamp.opencl.demos.bitonicsort;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.Map;
import java.util.Random;

import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static com.jogamp.opencl.CLProgram.*;

/**
 * Bitonic sort optimized for GPUs.
 * Uses NVIDIA's bitonic merge sort kernel.
 * @author Michael Bien
 */
public class BitonicSort {

    private static final String BITONIC_MERGE_GLOBAL = "bitonicMergeGlobal";
    private static final String BITONIC_MERGE_LOCAL  = "bitonicMergeLocal";
    private static final String BITONIC_SORT_LOCAL   = "bitonicSortLocal";
    private static final String BITONIC_SORT_LOCAL1  = "bitonicSortLocal1";

    private final static int LOCAL_SIZE_LIMIT = 1024;
    private final Map<String, CLKernel> kernels;

    public BitonicSort() throws IOException {

        final int sortDir  = 1;
        final int elements = 1048576;
        final int maxvalue = 1000000;

        out.println("Initializing OpenCL...");

        //Create the context
        CLContext context = null;

        try{

            context = CLContext.create();
            CLCommandQueue queue = context.getMaxFlopsDevice().createCommandQueue();

            out.println("Initializing OpenCL bitonic sorter...");
            kernels = initBitonicSort(queue);

            out.println("Creating OpenCL memory objects...");
            CLBuffer<IntBuffer> keyBuffer = context.createIntBuffer(elements, READ_ONLY, USE_BUFFER);
            System.out.println(keyBuffer.getCLSize()/1000000.0f);

            out.println("Initializing data...\n");
            Random random = new Random();
            for (int i = 0; i < elements; i++) {
                int rnd = random.nextInt(maxvalue);
                keyBuffer.getBuffer().put(i, rnd);
            }

            int arrayLength = elements;
            int batch = elements / arrayLength;

            out.printf("Test array length %d (%d arrays in the batch)...\n", arrayLength, batch);

            long time = currentTimeMillis();

            bitonicSort(queue, keyBuffer, keyBuffer, batch, arrayLength, sortDir);
            queue.putReadBuffer(keyBuffer, true);

            out.println(currentTimeMillis() - time+"ms");

            IntBuffer keys = keyBuffer.getBuffer();
            printSnapshot(keys, 20);
            checkIfSorted(keys);

            out.println("\nTEST PASSED");
        
        }finally{
            if(context!=null) {
                context.release();
            }
        }

    }
    
    private Map<String, CLKernel> initBitonicSort(CLCommandQueue queue) throws IOException {

        out.println("    creating bitonic sort program");

        CLContext context = queue.getContext();

        CLProgram program = context.createProgram(getClass().getResourceAsStream("BitonicSort.cl"))
                                   .build(define("LOCAL_SIZE_LIMIT", LOCAL_SIZE_LIMIT));

        Map<String, CLKernel> kernelMap = program.createCLKernels();

        out.println("    checking minimum supported workgroup size");
        //Check for work group size
        CLDevice device = queue.getDevice();
        long szBitonicSortLocal  = kernelMap.get(BITONIC_SORT_LOCAL).getWorkGroupSize(device);
        long szBitonicSortLocal1 = kernelMap.get(BITONIC_SORT_LOCAL1).getWorkGroupSize(device);
        long szBitonicMergeLocal = kernelMap.get(BITONIC_MERGE_LOCAL).getWorkGroupSize(device);

        if (    (szBitonicSortLocal < (LOCAL_SIZE_LIMIT / 2))
             || (szBitonicSortLocal1 < (LOCAL_SIZE_LIMIT / 2))
             || (szBitonicMergeLocal < (LOCAL_SIZE_LIMIT / 2))  ) {
            throw new RuntimeException("Minimum work-group size "+LOCAL_SIZE_LIMIT/2
                    +" required by this application is not supported on this device.");
        }

        return kernelMap;

    }

    public void bitonicSort(CLCommandQueue queue, CLBuffer<?> dstKey, CLBuffer<?> srcKey, int batch, int arrayLength, int dir) {

        if (arrayLength < 2) {
            throw new IllegalArgumentException("arrayLength was "+arrayLength);
        }

        // TODO Only power-of-two array lengths are supported so far

        dir = (dir != 0) ? 1 : 0;

        CLKernel sortlocal1  = kernels.get(BITONIC_SORT_LOCAL1);
        CLKernel sortlocal   = kernels.get(BITONIC_SORT_LOCAL);
        CLKernel mergeGlobal = kernels.get(BITONIC_MERGE_GLOBAL);
        CLKernel mergeLocal  = kernels.get(BITONIC_MERGE_LOCAL);

        if (arrayLength <= LOCAL_SIZE_LIMIT) {

            //        oclCheckError( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0, shrTRUE );

            //Launch bitonicSortLocal
            sortlocal.putArgs(dstKey, srcKey)
                     .putArg(arrayLength).putArg(dir).rewind();

            int localWorkSize = LOCAL_SIZE_LIMIT / 2;
            int globalWorkSize = batch * arrayLength / 2;
            queue.put1DRangeKernel(sortlocal, 0, globalWorkSize, localWorkSize);

        } else {

            //Launch bitonicSortLocal1
            sortlocal1.setArgs(dstKey, srcKey);

            int localWorkSize = LOCAL_SIZE_LIMIT / 2;
            int globalWorkSize = batch * arrayLength / 2;

            queue.put1DRangeKernel(sortlocal1, 0, globalWorkSize, localWorkSize);

            for (int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1) {
                for (int stride = size / 2; stride > 0; stride >>= 1) {
                    if (stride >= LOCAL_SIZE_LIMIT) {
                        //Launch bitonicMergeGlobal
                        mergeGlobal.putArgs(dstKey, dstKey)
                                   .putArg(arrayLength).putArg(size).putArg(stride).putArg(dir).rewind();

                        globalWorkSize = batch * arrayLength / 2;
                        queue.put1DRangeKernel(mergeGlobal, 0, globalWorkSize, 0);
                    } else {
                        //Launch bitonicMergeLocal
                        mergeLocal.putArgs(dstKey, dstKey)
                                  .putArg(arrayLength).putArg(stride).putArg(size).putArg(dir).rewind();

                        localWorkSize = LOCAL_SIZE_LIMIT / 2;
                        globalWorkSize = batch * arrayLength / 2;

                        queue.put1DRangeKernel(mergeLocal, 0, globalWorkSize, localWorkSize);
                        break;
                    }
                }
            }
        }
    }

    private void printSnapshot(IntBuffer buffer, int snapshot) {
        for(int i = 0; i < snapshot; i++)
            out.print(buffer.get() + ", ");
        out.println("...; " + buffer.remaining() + " more");
        buffer.rewind();
    }

    private void checkIfSorted(IntBuffer keys) {
        for (int i = 1; i < keys.capacity(); i++) {
            if (keys.get(i - 1) > keys.get(i)) {
                throw new RuntimeException("not sorted "+ keys.get(i - 1) +"!> "+ keys.get(i));
            }
        }
    }

    public static void main(String[] args) throws IOException {
        new BitonicSort();
    }
}
