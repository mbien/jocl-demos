/*
 * 20:48 Sunday, February 28 2010
 */

package com.jogamp.opencl.demos.radixsort;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.util.CLPlatformFilters;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.Random;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.System.*;
import static com.jogamp.opencl.CLDevice.Type.*;

/**
 * GPU radix sort demo.
 * @author Michael Bien
 */
public class RadixSortDemo {

    public RadixSortDemo() throws IOException {

        CLContext context = null;
        try{
            CLPlatform platform = CLPlatform.getDefault(CLPlatformFilters.type(GPU));
            if (platform == null) {
                throw new RuntimeException("this demo requires a GPU OpenCL implementation");
            }
            
            //single GPU setup
            context = CLContext.create(platform.getMaxFlopsDevice());
            CLCommandQueue queue = context.getDevices()[0].createCommandQueue();

            int maxValue = Integer.MAX_VALUE;
            int samples  = 10;

            int[] workgroupSizes = new int[] {128, 256};

            int[] runs = new int[] {   32768,
                                       65536,
                                      131072,
                                      262144,
                                      524288,
                                     1048576,
                                     2097152,
                                     4194304,
                                     8388608 };

            for (int i = 0; i < workgroupSizes.length; i++) {

                int workgroupSize = workgroupSizes[i];

                out.println("\n = = = workgroup size: "+workgroupSize+" = = = ");

                for(int run = 0; run < runs.length; run++) {

                    if(  workgroupSize==128 && runs[run] >= 8388608
                      || workgroupSize==256 && runs[run] <= 32768) {
                        continue; // we can only sort up to 4MB with wg size of 128
                    }

                    int numElements = runs[run];

                    CLBuffer<IntBuffer> array = context.createIntBuffer(numElements, READ_WRITE);
                    out.print("array size: " + array.getCLSize()/1000000.0f+"MB; ");
                    out.println("elements: " + array.getCLCapacity()/1000+"K");

                    fillBuffer(array, maxValue);

                    RadixSort radixSort = new RadixSort(queue, numElements, workgroupSize);
                    for(int a = 0; a < samples; a++) {

                        queue.finish();

                        long time = nanoTime();

                        queue.putWriteBuffer(array, false);
                        radixSort.sort(array, numElements, 32);
                        queue.putReadBuffer(array, true);

                        out.println("time: " + (nanoTime() - time)/1000000.0f+"ms");
                    }

                    out.print("snapshot: ");
                    printSnapshot(array.getBuffer(), 20);

                    out.println("validating...");
                    checkIfSorted(array.getBuffer());
                    out.println("values sorted");

                    array.release();
                    radixSort.release();
                }
            }

        }finally{
            if(context != null) {
                context.release();
            }
        }

    }

    private void fillBuffer(CLBuffer<IntBuffer> array, int maxValue) {
        Random random = new Random(42);
        for (int n = 0; n < array.getBuffer().capacity(); n++) {
            int rnd = random.nextInt(maxValue);
            array.getBuffer().put(n, rnd);
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
                throw new RuntimeException("not sorted "+ keys.get(i - 1) +" !> "+ keys.get(i));
            }
        }
    }

    public static void main(String[] args) throws IOException {
        new RadixSortDemo();
    }
}
