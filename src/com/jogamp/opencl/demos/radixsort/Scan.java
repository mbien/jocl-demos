/*
 * 22:12 Sunday, February 28 2010
 */
package com.jogamp.opencl.demos.radixsort;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLResource;
import java.io.IOException;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static com.jogamp.opencl.CLProgram.CompilerOptions.*;

/**
 *
 * @author Michael Bien
 */
public class Scan implements CLResource {

    private final static int MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE = 1024;
    private final static int MAX_LOCAL_GROUP_SIZE = 256;
    private final static int WORKGROUP_SIZE = 256;
    private final static int MAX_BATCH_ELEMENTS = 64 * 1048576;
    private final static int MIN_SHORT_ARRAY_SIZE = 4;
    private final static int MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
    private final static int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
    private final static int MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

    private final CLKernel ckScanExclusiveLocal1;
    private final CLKernel ckScanExclusiveLocal2;
    private final CLKernel ckUniformUpdate;

    private final CLCommandQueue queue;
    private final CLProgram program;
    private CLBuffer<?> buffer;

    public Scan(CLCommandQueue queue, int numElements) throws IOException {

        this.queue = queue;

        CLContext context = queue.getContext();
        if (numElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE) {
            buffer = context.createBuffer(numElements / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * 4, READ_WRITE);
        }
        program = context.createProgram(getClass().getResourceAsStream("Scan_b.cl"))
                         .build(ENABLE_MAD);

        ckScanExclusiveLocal1 = program.createCLKernel("scanExclusiveLocal1");
        ckScanExclusiveLocal2 = program.createCLKernel("scanExclusiveLocal2");
        ckUniformUpdate       = program.createCLKernel("uniformUpdate");
    }

    // main exclusive scan routine
    void scanExclusiveLarge(CLBuffer<?> dst, CLBuffer<?> src, int batchSize, int arrayLength) {

        //Check power-of-two factorization
        if(!isPowerOf2(arrayLength)) {
            throw new RuntimeException();
        }

        //Check supported size range
        if (!((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE))) {
            throw new RuntimeException();
        }

        //Check total batch size limit
        if (!((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS)) {
            throw new RuntimeException();
        }

        scanExclusiveLocal1(dst, src, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE), 4 * WORKGROUP_SIZE);
        scanExclusiveLocal2(buffer, dst, src, batchSize, arrayLength / (4 * WORKGROUP_SIZE));
        uniformUpdate(dst, buffer, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
    }

    void scanExclusiveLocal1(CLBuffer<?> dst, CLBuffer<?> src, int n, int size) {

        ckScanExclusiveLocal1.putArg(dst).putArg(src).putArgSize(2 * WORKGROUP_SIZE * 4).putArg(size)
                             .rewind();

        int localWorkSize = WORKGROUP_SIZE;
        int globalWorkSize = (n * size) / 4;

        queue.put1DRangeKernel(ckScanExclusiveLocal1, 0, globalWorkSize, localWorkSize);
    }

    void scanExclusiveLocal2(CLBuffer<?> buffer, CLBuffer<?> dst, CLBuffer<?> src, int n, int size) {

        int elements = n * size;
        ckScanExclusiveLocal2.putArg(buffer).putArg(dst).putArg(src).putArgSize(2 * WORKGROUP_SIZE * 4)
                             .putArg(elements).putArg(size).rewind();

        int localWorkSize = WORKGROUP_SIZE;
        int globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);

        queue.put1DRangeKernel(ckScanExclusiveLocal2, 0, globalWorkSize, localWorkSize);
    }

    void uniformUpdate(CLBuffer<?> dst, CLBuffer<?> buffer, int n) {

        ckUniformUpdate.setArgs(dst, buffer);

        int localWorkSize  = WORKGROUP_SIZE;
        int globalWorkSize = n * WORKGROUP_SIZE;

        queue.put1DRangeKernel(ckUniformUpdate, 0, globalWorkSize, localWorkSize);
    }

    private int iSnapUp(int dividend, int divisor) {
        return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
    }

    public static boolean isPowerOf2(int x) {
        return ((x - 1) & x) == 0;
    }

    public void release() {
        program.release();

        if(buffer!=null) {
            buffer.release();
        }
    }

    @Override
    public boolean isReleased() {
        return program.isReleased();
    }

    public void close() {
        release();
    }
}
