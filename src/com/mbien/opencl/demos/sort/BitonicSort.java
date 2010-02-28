/*
 * 18:42 Saturday, February 27 2010
 */
package com.mbien.opencl.demos.sort;

import com.mbien.opencl.CLBuffer;
import com.mbien.opencl.CLCommandQueue;
import com.mbien.opencl.CLContext;
import com.mbien.opencl.CLDevice;
import com.mbien.opencl.CLKernel;
import com.mbien.opencl.CLProgram;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.Map;
import java.util.Random;

import static java.lang.System.*;
import static com.mbien.opencl.CLMemory.Mem.*;
import static com.mbien.opencl.CLProgram.*;

/**
 * Bitonic sort optimized for GPUs.
 * Uses NVIDIA's bitonic merge sort kernel.
 * @author Michael Bien
 */
public class BitonicSort {

    private static final String BITONIC_MERGE_LOCAL = "bitonicMergeLocal";
    private static final String BITONIC_SORT_LOCAL  = "bitonicSortLocal";
    private static final String BITONIC_SORT_LOCAL1 = "bitonicSortLocal1";

    private final static int LOCAL_SIZE_LIMIT = 1024;
    private final Map<String, CLKernel> kernels;

    public BitonicSort() throws IOException {

        final int sortDir  = 1;
        final int elements = 1024;
        final int maxvalue = 1000000000;

        System.out.println("Initializing OpenCL...");

        //Create the context
        CLContext context = CLContext.create();
        CLCommandQueue queue = context.getMaxFlopsDevice().createCommandQueue();

        System.out.println("Initializing OpenCL bitonic sorter...");
        kernels = initBitonicSort(context, queue);


        System.out.println("Creating OpenCL memory objects...");
        CLBuffer<IntBuffer> keyBuffer = context.createIntBuffer(elements, READ_ONLY, USE_BUFFER);

        // in case of key/value pairs
//        CLBuffer<IntBuffer> valueBuffer  = context.createIntBuffer(elements, READ_ONLY, USE_BUFFER);

        System.out.println("Initializing data...\n");
        Random random = new Random();
        for (int i = 0; i < elements; i++) {
            int rnd = random.nextInt(maxvalue);
            keyBuffer.getBuffer().put(i, rnd);
//            valueBuffer.getBuffer().put(i, rnd); // value can be arbitary
        }

        int arrayLength = elements;
        int batch = elements / arrayLength;

        System.out.printf("Test array length %d (%d arrays in the batch)...\n", arrayLength, batch);

//            long time = System.currentTimeMillis();

        bitonicSort(queue, keyBuffer, batch, arrayLength, sortDir);

        queue.putReadBuffer(keyBuffer, true);
//        queue.putReadBuffer(valueBuffer, true);
//            System.out.println(System.currentTimeMillis() - time);

        IntBuffer keys = keyBuffer.getBuffer();
        printSnapshot(keys, 10);
        checkIfSorted(keys);

//        IntBuffer values = valueBuffer.getBuffer();
//        printSnapshot(values, 10);
//        checkIfSorted(values);

        System.out.println();

        System.out.println("TEST PASSED");
        
        context.release();

    }
    
    private Map<String, CLKernel> initBitonicSort(CLContext context, CLCommandQueue queue) throws IOException {

        System.out.println("    creating bitonic sort program");

        CLProgram program = context.createProgram(getClass().getResourceAsStream("BitonicSort.cl"))
                                   .build(define("LOCAL_SIZE_LIMIT", LOCAL_SIZE_LIMIT));

        Map<String, CLKernel> kernels = program.createCLKernels();

        System.out.println("    checking minimum supported workgroup size");
        //Check for work group size
        CLDevice device = queue.getDevice();
        long szBitonicSortLocal  = kernels.get(BITONIC_SORT_LOCAL).getWorkGroupSize(device);
        long szBitonicSortLocal1 = kernels.get(BITONIC_SORT_LOCAL1).getWorkGroupSize(device);
        long szBitonicMergeLocal = kernels.get(BITONIC_MERGE_LOCAL).getWorkGroupSize(device);

        if (    (szBitonicSortLocal < (LOCAL_SIZE_LIMIT / 2))
             || (szBitonicSortLocal1 < (LOCAL_SIZE_LIMIT / 2))
             || (szBitonicMergeLocal < (LOCAL_SIZE_LIMIT / 2))  ) {
            throw new RuntimeException("Minimum work-group size "+LOCAL_SIZE_LIMIT/2
                    +" required by this application is not supported on this device.");
        }

        return kernels;

    }

    public void bitonicSort(CLCommandQueue queue, CLBuffer<?> keys, int batch, int arrayLength, int dir) {
        this.bitonicSort(queue, keys, keys, keys, keys, batch, arrayLength, dir);
    }

    public void bitonicSort(CLCommandQueue queue, CLBuffer<?> keys, CLBuffer<?> values, int batch, int arrayLength, int dir) {
        this.bitonicSort(queue, keys, values, keys, values, batch, arrayLength, dir);
    }

    public void bitonicSort(CLCommandQueue queue, CLBuffer<?> dstKey, CLBuffer<?> dstVal, CLBuffer<?> srcKey, CLBuffer<?> srcVal, int batch, int arrayLength, int dir) {

        if (arrayLength < 2) {
            throw new IllegalArgumentException("arrayLength was "+arrayLength);
        }

        // TODO Only power-of-two array lengths are supported so far

        dir = (dir != 0) ? 1 : 0;

        if (arrayLength <= LOCAL_SIZE_LIMIT) {

            //        oclCheckError( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0, shrTRUE );

            //Launch bitonicSortLocal
            CLKernel kernel = kernels.get(BITONIC_SORT_LOCAL)
                    .putArgs(dstKey, dstVal, srcKey, srcVal)
                    .putArg(arrayLength).putArg(dir).rewind();

            int localWorkSize = LOCAL_SIZE_LIMIT / 2;
            int globalWorkSize = batch * arrayLength / 2;
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);

        } else {

            //Launch bitonicSortLocal1
            CLKernel kernel = kernels.get(BITONIC_SORT_LOCAL1)
                    .setArgs(dstKey, dstVal, srcKey, srcVal);

            int localWorkSize = LOCAL_SIZE_LIMIT / 2;
            int globalWorkSize = batch * arrayLength / 2;

            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);

            for (int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1) {
                for (int stride = size / 2; stride > 0; stride >>= 1) {
                    if (stride >= LOCAL_SIZE_LIMIT) {
                        //Launch bitonicMergeGlobal
                        kernel = kernels.get("bitonicMergeGlobal")
                                .putArgs(dstKey, dstVal, dstKey, dstVal)
                                .putArg(arrayLength).putArg(size).putArg(stride).putArg(dir).rewind();

                        globalWorkSize = batch * arrayLength / 2;
                        queue.put1DRangeKernel(kernel, 0, globalWorkSize, 0);
                    } else {
                        //Launch bitonicMergeLocal
                        kernel = kernels.get(BITONIC_MERGE_LOCAL)
                                .putArgs(dstKey, dstVal, dstKey, dstVal)
                                .putArg(arrayLength).putArg(stride).putArg(size).putArg(dir).rewind();

                        localWorkSize = LOCAL_SIZE_LIMIT / 2;
                        globalWorkSize = batch * arrayLength / 2;

                        queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
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
