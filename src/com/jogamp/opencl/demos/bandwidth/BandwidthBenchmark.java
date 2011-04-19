/*
 * Created on Tuesday, September 14 2010 17:19
 */

package com.jogamp.opencl.demos.bandwidth;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLPlatform;

import static com.jogamp.opencl.CLMemory.Map.*;
import com.jogamp.opencl.CLMemory.Mem;
import static com.jogamp.opencl.CLMemory.Mem.*;

import java.nio.ByteBuffer;

/**
 * Port of Nvidia's BandwidthTest to JOCL HLB.
 * @author Michael Bien
 */
public class BandwidthBenchmark {

    // defines, project
    private static int MEMCOPY_ITERATIONS = 100;
    private static int DEFAULT_SIZE = (32 * (1 << 20));    //32 M
    private static int DEFAULT_INCREMENT = (1 << 22);     //4 M
    private static int CACHE_CLEAR_SIZE = (1 << 24);       //16 M

    //shmoo mode defines
    private static int SHMOO_MEMSIZE_MAX = (1 << 26);         //64 M
    private static int SHMOO_MEMSIZE_START = (1 << 10);         //1 KB
    private static int SHMOO_INCREMENT_1KB = (1 << 10);         //1 KB
    private static int SHMOO_INCREMENT_2KB = (1 << 11);         //2 KB
    private static int SHMOO_INCREMENT_10KB = (10 * (1 << 10));  //10KB
    private static int SHMOO_INCREMENT_100KB = (100 * (1 << 10)); //100 KB
    private static int SHMOO_INCREMENT_1MB = (1 << 20);         //1 MB
    private static int SHMOO_INCREMENT_2MB = (1 << 21);         //2 MB
    private static int SHMOO_INCREMENT_4MB = (1 << 22);         //4 MB
    private static int SHMOO_LIMIT_20KB = (20 * (1 << 10));  //20 KB
    private static int SHMOO_LIMIT_50KB = (50 * (1 << 10));  //50 KB
    private static int SHMOO_LIMIT_100KB = (100 * (1 << 10)); //100 KB
    private static int SHMOO_LIMIT_1MB = (1 << 20);         //1 MB
    private static int SHMOO_LIMIT_16MB = (1 << 24);         //16 MB
    private static int SHMOO_LIMIT_32MB = (1 << 25);         //32 MB

    private enum TEST_MODE { QUICK, RANGE, SHMOO };
    private enum COPY { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
    private enum MEMORY { PAGEABLE, PINNED };
    private enum ACCESS { MAPPED, DIRECT };


    public static void main(String[] args) {

        int start = DEFAULT_SIZE;
        int end = DEFAULT_SIZE;
        int increment = DEFAULT_INCREMENT;

        TEST_MODE mode = TEST_MODE.QUICK;
        MEMORY memMode = MEMORY.PAGEABLE;
        ACCESS accMode = ACCESS.DIRECT;

        CLPlatform[] platforms = CLPlatform.listCLPlatforms();
        CLPlatform platform = platforms[0];

        // prefere NV
        for (CLPlatform p : platforms) {
            if(p.getICDSuffix().equals("NV")) {
                platform = p;
                break;
            }
        }

        CLDevice device = platform.getMaxFlopsDevice();

        int deviceIndex = -1;
        for (String arg : args) {
            if(arg.startsWith("--access=")) {
                accMode = ACCESS.valueOf(arg.substring(9).toUpperCase());
            }else if(arg.startsWith("--memory=")) {
                memMode = MEMORY.valueOf(arg.substring(9).toUpperCase());
            }else if(arg.startsWith("--device=")) {
                deviceIndex = Integer.parseInt(arg.substring(9).toUpperCase());
            }else if(arg.startsWith("--mode=")) {
                mode = TEST_MODE.valueOf(arg.substring(7).toUpperCase());
            }else if(arg.startsWith("--platform=")) {
                platform = platforms[Integer.parseInt(arg.substring(11))];
            }else{
                System.out.println("unknown arg: "+arg);
                System.exit(1);
            }
        }
        if(deviceIndex != -1) {
            device = platform.listCLDevices()[deviceIndex];
        }

        CLContext context = CLContext.create(device);

        System.out.println();
        System.out.println(platform);
        System.out.println(context);
        System.out.println();

        // Run tests
        testBandwidth(context, start, end, increment, mode, COPY.HOST_TO_DEVICE, accMode, memMode);
        testBandwidth(context, start, end, increment, mode, COPY.DEVICE_TO_HOST, accMode, memMode);
        testBandwidth(context, start, end, increment, mode, COPY.DEVICE_TO_DEVICE, accMode, memMode);

        context.release();
    }

    private static void testBandwidth(CLContext context, int start, int end, int increment, TEST_MODE mode, COPY kind, ACCESS accMode, MEMORY memMode) {
        switch (mode) {
            case QUICK:
                testBandwidthQuick(context, DEFAULT_SIZE, kind, accMode, memMode);
                break;
            case RANGE:
                testBandwidthRange(context, start, end, increment, kind, accMode, memMode);
                break;
            case SHMOO:
                testBandwidthShmoo(context, kind, accMode, memMode);
                break;
            default:
                break;
        }
    }

    /**
     * Run a quick mode bandwidth test
     */
    private static void testBandwidthQuick(CLContext context, int size, COPY kind, ACCESS accMode, MEMORY memMode) {
        testBandwidthRange(context, size, size, DEFAULT_INCREMENT, kind, accMode, memMode);
    }

    /**
     * Run a range mode bandwidth test
     */
    private static void testBandwidthRange(CLContext context, int start, int end, int increment, COPY kind, ACCESS accMode, MEMORY memMode) {
        //count the number of copies we're going to run
        int count = 1 + ((end - start) / increment);

        int[] memSizes = new int[count];
        double[] bandwidths = new double[count];

        // Use the device asked by the user
        CLDevice[] devices = context.getDevices();
        for (CLDevice device : devices) {
            CLCommandQueue queue = device.createCommandQueue();

            //run each of the copies
            for (int i = 0; i < count; i++) {
                memSizes[i] = start + i * increment;
                switch (kind) {
                    case DEVICE_TO_HOST:
                        bandwidths[i] += testDeviceToHostTransfer(queue, memSizes[i], accMode, memMode);
                        break;
                    case HOST_TO_DEVICE:
                        bandwidths[i] += testHostToDeviceTransfer(queue, memSizes[i], accMode, memMode);
                        break;
                    case DEVICE_TO_DEVICE:
                        bandwidths[i] += testDeviceToDeviceTransfer(queue, memSizes[i]);
                        break;
                }
            }
            queue.release();
        }

        //print results
        printResultsReadable(memSizes, bandwidths, count, kind, accMode, memMode, count);
    }

    /**
     *  Intense shmoo mode - covers a large range of values with varying increments
     */
    private static void testBandwidthShmoo(CLContext context, COPY kind, ACCESS accMode, MEMORY memMode) {

        //count the number of copies to make
        int count = 1 + (SHMOO_LIMIT_20KB / SHMOO_INCREMENT_1KB)
                + ((SHMOO_LIMIT_50KB - SHMOO_LIMIT_20KB) / SHMOO_INCREMENT_2KB)
                + ((SHMOO_LIMIT_100KB - SHMOO_LIMIT_50KB) / SHMOO_INCREMENT_10KB)
                + ((SHMOO_LIMIT_1MB - SHMOO_LIMIT_100KB) / SHMOO_INCREMENT_100KB)
                + ((SHMOO_LIMIT_16MB - SHMOO_LIMIT_1MB) / SHMOO_INCREMENT_1MB)
                + ((SHMOO_LIMIT_32MB - SHMOO_LIMIT_16MB) / SHMOO_INCREMENT_2MB)
                + ((SHMOO_MEMSIZE_MAX - SHMOO_LIMIT_32MB) / SHMOO_INCREMENT_4MB);

        int[] memSizes = new int[count];
        double[] bandwidths = new double[count];

        // Use the device asked by the user
        CLDevice[] devices = context.getDevices();
        for (CLDevice device : devices) {
            // Allocate command queue for the device
            CLCommandQueue queue = device.createCommandQueue();

            //Run the shmoo
            int iteration = 0;
            int memSize = 0;
            while (memSize <= SHMOO_MEMSIZE_MAX) {
                if (memSize < SHMOO_LIMIT_20KB) {
                    memSize += SHMOO_INCREMENT_1KB;
                } else if (memSize < SHMOO_LIMIT_50KB) {
                    memSize += SHMOO_INCREMENT_2KB;
                } else if (memSize < SHMOO_LIMIT_100KB) {
                    memSize += SHMOO_INCREMENT_10KB;
                } else if (memSize < SHMOO_LIMIT_1MB) {
                    memSize += SHMOO_INCREMENT_100KB;
                } else if (memSize < SHMOO_LIMIT_16MB) {
                    memSize += SHMOO_INCREMENT_1MB;
                } else if (memSize < SHMOO_LIMIT_32MB) {
                    memSize += SHMOO_INCREMENT_2MB;
                } else {
                    memSize += SHMOO_INCREMENT_4MB;
                }

                memSizes[iteration] = memSize;
                switch (kind) {
                    case DEVICE_TO_HOST:
                        bandwidths[iteration] += testDeviceToHostTransfer(queue, memSizes[iteration], accMode, memMode);
                        break;
                    case HOST_TO_DEVICE:
                        bandwidths[iteration] += testHostToDeviceTransfer(queue, memSizes[iteration], accMode, memMode);
                        break;
                    case DEVICE_TO_DEVICE:
                        bandwidths[iteration] += testDeviceToDeviceTransfer(queue, memSizes[iteration]);
                        break;
                }
                iteration++;
            }
            queue.release();
        }

        //print results
        printResultsReadable(memSizes, bandwidths, count, kind, accMode, memMode, count);

    }

    /**
     *  test the bandwidth of a device to host memcopy of a specific size
     */
    private static double testDeviceToHostTransfer(CLCommandQueue queue, int memSize, ACCESS accMode, MEMORY memMode) {

        ByteBuffer h_data = null;
        CLBuffer<?> cmPinnedData = null;
        CLBuffer<?> cmDevData;

        CLContext context = queue.getContext();

        //allocate and init host memory, pinned or conventional
        if (memMode == memMode.PINNED) {
            // Create a host buffer
            cmPinnedData = context.createBuffer(memSize, Mem.READ_WRITE, Mem.ALLOCATE_BUFFER);

            // Get a mapped pointer
            h_data = queue.putMapBuffer(cmPinnedData, WRITE, true);
            fill(h_data);

            // unmap and make data in the host buffer valid
            queue.putUnmapMemory(cmPinnedData, h_data);
        } else { // PAGED
            // standard host alloc
            h_data = Buffers.newDirectByteBuffer(memSize);
            fill(h_data);
        }

        // allocate device memory
        cmDevData = context.createBuffer(memSize, Mem.READ_WRITE);

        // initialize device memory
        if (memMode == memMode.PINNED) {
            // Get a mapped pointer
            h_data = queue.putMapBuffer(cmPinnedData, WRITE, true);

            cmDevData = cmDevData.cloneWith(h_data);
            queue.putWriteBuffer(cmDevData, false);
        } else { // PAGED
            cmDevData = cmDevData.cloneWith(h_data);
            queue.putWriteBuffer(cmDevData, false);
        }

        // Sync queue to host, start timer 0, and copy data from GPU to Host
        queue.finish();
        
        long delta = System.nanoTime();

        if (accMode == accMode.DIRECT) {
            // DIRECT:  API access to device buffer
            cmDevData = cmDevData.cloneWith(h_data);
            for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
                queue.putReadBuffer(cmDevData, false);
            }
            queue.finish();
        } else {
            // MAPPED: mapped pointers to device buffer for conventional pointer access
            ByteBuffer dm_idata = queue.putMapBuffer(cmDevData, WRITE, true);
            for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
                h_data.put(dm_idata).rewind();
                dm_idata.rewind();
            }
            queue.putUnmapMemory(cmDevData, dm_idata);
        }

        //get the the elapsed time in seconds
        delta = System.nanoTime() - delta;

        //clean up memory
        cmDevData.release();

        if (cmPinnedData != null) {
            queue.putUnmapMemory(cmPinnedData, h_data);
            cmPinnedData.release();
        }

        //calculate bandwidth in MB/s
        double elapsedTime = delta/1000000000.0;
        return ((double) memSize * (double) MEMCOPY_ITERATIONS) / (elapsedTime*(double)(1 << 20));
    }

    /**
     *  test the bandwidth of a device to host memcopy of a specific size
     */
    private static double testHostToDeviceTransfer(CLCommandQueue queue, int memSize, ACCESS accMode, MEMORY memMode) {

        ByteBuffer h_data;
        CLBuffer<?> cmPinnedData = null;
        CLBuffer<?> cmDevData;

        CLContext context = queue.getContext();

        // Allocate and init host memory, pinned or conventional
        if (memMode == memMode.PINNED) {
            // Create a host buffer
            cmPinnedData = context.createBuffer(memSize, Mem.READ_WRITE, Mem.ALLOCATE_BUFFER);

            // Get a mapped pointer
            h_data = queue.putMapBuffer(cmPinnedData, WRITE, true);

            //initialize
            fill(h_data);

            // unmap and make data in the host buffer valid
            queue.putUnmapMemory(cmPinnedData, h_data);
        } else { // PAGED
            // standard host alloc
            h_data = Buffers.newDirectByteBuffer(memSize);
            fill(h_data);
        }

        // allocate device memory
        cmDevData = context.createBuffer(memSize, Mem.READ_WRITE);

        // Sync queue to host, start timer 0, and copy data from Host to GPU
        queue.finish();

        long delta = System.nanoTime();

        if (accMode == accMode.DIRECT) {
            if (memMode == memMode.PINNED) {
                // Get a mapped pointer
                h_data = queue.putMapBuffer(cmPinnedData, WRITE, true);
            }

            // DIRECT:  API access to device buffer
            cmDevData = cmDevData.cloneWith(h_data);
            for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
                queue.putWriteBuffer(cmDevData, false);
            }
            queue.finish();
        } else {

            // MAPPED: mapped pointers to device buffer and conventional pointer access
            ByteBuffer dm_idata = queue.putMapBuffer(cmDevData, READ, true);
            for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
                dm_idata.put(h_data).rewind();
                h_data.rewind();
            }
            queue.putUnmapMemory(cmDevData, dm_idata);
        }

        //get the the elapsed time in ms
        delta = System.nanoTime() - delta;

        //clean up memory
        cmDevData.release();

        if (cmPinnedData != null) {
//            cmPinnedData = cmPinnedData.cloneWith(h_data);
//            queue.putUnmapMemory(cmPinnedData);
            cmPinnedData.release();
        }

        //calculate bandwidth in MB/s
        double elapsedTime = delta/1000000000.0;
        return ((double) memSize * (double) MEMCOPY_ITERATIONS) / (elapsedTime*(double)(1 << 20));
    }

    /**
     *  test the bandwidth of a device to host memcopy of a specific size
     */
    private static double testDeviceToDeviceTransfer(CLCommandQueue queue, int memSize) {

        CLContext context = queue.getContext();

        //allocate host memory
        ByteBuffer h_idata = Buffers.newDirectByteBuffer(memSize);
        fill(h_idata);

        // allocate device input and output memory and initialize the device input memory
        CLBuffer<?> d_idata = context.createBuffer(memSize, READ_ONLY);
        CLBuffer<?> d_odata = context.createBuffer(memSize, WRITE_ONLY);

        d_idata = d_idata.cloneWith(h_idata);
        queue.putWriteBuffer(d_idata, true);

        // Sync queue to host, start timer 0, and copy data from one GPU buffer to another GPU bufffer
        queue.finish();

        long delta = System.nanoTime();

        for (int i = 0; i < MEMCOPY_ITERATIONS; i++) {
            queue.putCopyBuffer(d_idata, d_odata);
        }

        // Sync with GPU
        queue.finish();

        //get the the elapsed time in ms
        delta = System.nanoTime() - delta;

        //clean up memory on host and device
        d_idata.release();
        d_odata.release();

        // Calculate bandwidth in MB/s
        //      This is for kernels that read and write GMEM simultaneously
        //      Obtained Throughput for unidirectional block copies will be 1/2 of this #
        double elapsedTime = delta/1000000000.0;
        return 2.0 * ((double) memSize * (double) MEMCOPY_ITERATIONS) / (elapsedTime*(double)(1 << 20));
    }

    private static void fill(ByteBuffer buffer) {
        int i = 0;
        while(buffer.remaining() > 0) {
            buffer.putChar((char) (i++ & 0xff));
        }
        buffer.rewind();
    }

    /**
     * print results in an easily read format
     */
    private static void printResultsReadable(int[] memSizes, double[] bandwidths, int count, COPY kind, ACCESS accMode, MEMORY memMode, int iNumDevs) {
        // log config information
        if (kind == COPY.DEVICE_TO_DEVICE) {
            System.out.print("Device to Device Bandwidth, "+iNumDevs+" Device(s), ");
        } else {
            if (kind == COPY.DEVICE_TO_HOST) {
                System.out.print("Device to Host Bandwidth, "+iNumDevs+" Device(s), ");
            } else if (kind == COPY.HOST_TO_DEVICE) {
                System.out.print("Host to Device Bandwidth, "+iNumDevs+" Device(s), ");
            }
            if (memMode == memMode.PAGEABLE) {
                System.out.print("Paged memory");
            } else if (memMode == memMode.PINNED) {
                System.out.print("Pinned memory");
            }
            if (accMode == accMode.DIRECT) {
                System.out.println(", direct access");
            } else if (accMode == accMode.MAPPED) {
                System.out.println(", mapped access");
            }
        }
        System.out.println();

        System.out.println("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
        int i;
        for (i = 0; i < (count - 1); i++) {
            System.out.printf("   %s\t\t\t%s%.1f\n", memSizes[i], (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
        }
        System.out.printf("   %s\t\t\t%s%.1f\n\n", memSizes[i], (memSizes[i] < 10000) ? "\t" : "", bandwidths[i]);
    }

}
