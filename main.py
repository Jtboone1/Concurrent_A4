import os
import sys
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

message_count = 0
debug_arg = ""

if len(sys.argv) > 1:
    debug_arg = sys.argv[1]

debug = True if debug_arg == "debug" else False

# Clear file
if debug:
    f = open("debug" + str(rank) + ".txt", "w").close()

# Each array is the rank of the process that will filter the image.
# We will filter 4 sub arrays concurrently. The other 12 will be responsible
# for providing the neighbor pixels.
order_arr = [
    [0, 6, 8, 14],
    [1, 7, 9, 15],
    [2, 4, 10, 12],
    [3, 5, 11, 13]
]

# This keeps track of each order of subimages we're filtering
order_idx = 0

sub_images = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]
]

rank_neighbors = [
    [1, 4, 5],                      # Rank 0 neighbors
    [0, 2, 4, 5, 6],                # Rank 1 neighbors
    [1, 3, 5, 6, 7],                # Rank 2 neighbors
    [2, 6, 7],                      # Rank 3 neighbors
    [0, 1, 5, 8, 9],                # Rank 4 neighbors
    [0, 1, 2, 4, 6, 8, 9, 10],      # Rank 5 neighbors
    [1, 2, 3, 5, 7, 9, 10, 11],     # Rank 6 neighbors
    [2, 3, 6, 10, 11],              # Rank 7 neighbors
    [4, 5, 9, 12, 13],              # Rank 8 neighbors
    [4, 5, 6, 8, 10, 12, 13, 14],   # Rank 9 neighbors
    [5, 6, 7, 9, 11, 13, 14, 15],   # Rank 10 neighbors
    [6, 7, 10, 14, 15],             # Rank 11 neighbors
    [8, 9, 13],                     # Rank 12 neighbors
    [8, 9, 10, 12, 14],             # Rank 13 neighbors
    [9, 10, 11, 13, 15],            # Rank 14 neighbors
    [10, 11, 14]                    # Rank 15 neighbors
]

gaussian_kernel = [
    [0, 0, 3,   2,   2,   2, 3, 0, 0],
    [0, 2, 3,   5,   5,   5, 3, 2, 0],
    [3, 3, 5,   3,   0,   3, 5, 3, 3],
    [2, 5, 3, -12, -23, -12, 3, 5, 2],
    [2, 5, 0, -23, -40, -23, 0, 5, 2],
    [2, 5, 3, -12, -23, -12, 3, 5, 2],
    [3, 3, 5,   3,   0,   3, 5, 3, 3],
    [0, 2, 3,   5,   5,   5, 3, 2, 0],
    [0, 0, 3,   2,   2,   2, 3, 0, 0],
]

def printr(msg, bypass=False):
    global message_count
    message_count += 1

    if debug and (message_count % 400 == 0 or bypass):
        f = open("debug" + str(rank) + ".txt", "a+")
        f.write("\n" + "Message (" + str(message_count) +  "): " + msg)
        f.close()


# Three Scenarios here when iterating over the gaussian matrix:
# A) The kernel fits within our sub image, in which case we can just filter normally,
# B) The kernel is outside the sub image but still inside the entire image, in which case we need to use MPI to get the value of the pixel that is outside our sub image bounds
# C) The kernel is outside the sub image AND outside the entire image, in which case we set that value to 0 (Or in this functions case, we just don't do anything)
def filter_pixel(x, y, min_x, min_y, max_x, max_y):

    filtered_pixel = 0

    for j in range(9):
        for i in range(9):
            image_x_offset = i - 4
            image_y_offset = j - 4

            image_x = x + image_x_offset
            image_y = y + image_y_offset

            if (image_x < min_x or image_x > max_x or image_y < min_y or image_y > max_y) and (image_x >= 0 and image_x < 256 and image_y >= 0 and image_y < 256):

                # If were outside our sub image bounds, we need to call our neighbor in order to retreive the correct pixel
                neighbor_x = 0
                neighbor_y = 0
                
                # Find neighbor that has the pixel were looking for
                for row in range(4):
                    for col in range(4):
                        if rank == sub_images[row][col]:
                            neighbor_x = col
                            neighbor_y = row

                if image_x < min_x:
                    neighbor_x -= 1
                elif image_x > max_x:
                    neighbor_x += 1

                if image_y < min_y:
                    neighbor_y -= 1
                elif image_y > max_y:
                    neighbor_y += 1

                printr("Requesting pixel from neighbor: " + str(sub_images[neighbor_y][neighbor_x]))
                req = comm.isend([image_y, image_x], dest=sub_images[neighbor_y][neighbor_x])
                req.wait()

                req = comm.irecv(source=sub_images[neighbor_y][neighbor_x])
                pixel_value = req.wait()

                filtered_pixel += gaussian_kernel[j][i] * pixel_value

            elif image_x >= 0 and image_x < 256 and image_y >= 0 and image_y < 256:
                filtered_pixel += gaussian_kernel[j][i] * original_image[image_y][image_x]

    if filtered_pixel > 255:
        filtered_pixel = 255

    if filtered_pixel < 0:
        filtered_pixel = 0

    return filtered_pixel

# Create a matrix of 0s and fill it for the original image
pixel_values_1d = []

f = open("pepper.ascii.pgm", "r")

lines = f.readlines()
line_count = 0

# Load original image from file
for line in lines:
    # Skip over first 4 lines in file
    if line_count > 3:
        pixel_values = line.split()
        for value in pixel_values:
            pixel_values_1d.append(int(value))
    else:
        line_count += 1

f.close()

# Convert 1D array to 256 x 256 2D array
original_image = np.reshape(pixel_values_1d, (256,256))

# Rank 16 will be responsible for reconstructing and writing to the PGM filtered file
if rank == 16:

    pixels = 256 * 256
    pixels_filtered = 0

    filtered_image = np.zeros((256, 256))

    # Well send a 3 item array, [pixel_x_coor, pixel_y_coor, pixel_filtered_value], and add it to the filtered array.
    while (pixels_filtered < pixels):
        req = comm.irecv(source=MPI.ANY_SOURCE)
        pixel_filtered_info = req.wait()
        filtered_image[pixel_filtered_info[1], pixel_filtered_info[0]] = pixel_filtered_info[2]
        pixels_filtered += 1

        printr(str(pixels_filtered) + "/65536")

    # Writing an output file
    # This will be the end of our program.
    f = open("output.pgm", "w")

    printr("OUTPUTTING: ")
    line = "P2 " + "\n" + "256 256 " + "\n" + "255" + "\n"

    max_number = 0
    for j in range(256):
        for i in range(256):
            
            if max_number > 16:
                line += "\n"
                max_number = 0

            if (max_number != 0):
                line += " "

            line += str(int(filtered_image[j][i]))

            max_number += 1

    line += "\n"
    f.write(line)
    f.close()

# Rank 17 will be the controller process
elif rank == 17: 

    while(order_idx != 4):

        # Start by setting each process type
        for curr_rank in range (16):
            if curr_rank in order_arr[order_idx]:
                printr("Setting " + str(curr_rank) + " to Filter", True)
                req = comm.isend("Filter", dest=curr_rank)
                req.wait()

            else:
                printr("Setting " + str(curr_rank) + " to Provide", True)
                req = comm.isend("Provide", dest=curr_rank)
                req.wait()


        done = 0
        done_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        printr("Will now wait for every Subimage to finish this layer", True)

        # When all kernels are done providing / filtering, we move to the next order
        while done != 16:
            for curr_rank, is_done in enumerate(done_arr):
                if is_done != 1:
                    printr("Waiting on rank: " + str(curr_rank), True)
                    req = comm.irecv(source=curr_rank)
                    req.wait()

                    done_arr[curr_rank] = 1
                    done += 1
                    printr(str(done) + " subimages finished in order layer: " + str(order_idx), True)

        order_idx += 1

else:
    while (order_idx != 4):

        req = comm.irecv(source=17)
        filter_or_provide = req.wait()

        # Get offsets for where they are in array
        # So sub image 5 will have offset x = 63 offset y = 63 for example
        offset_x = rank % 4 * 64
        offset_y = int(rank / 4) * 64

        if (offset_x > 0):
            offset_x - 1
        if (offset_y > 0):
            offset_y - 1

        if (filter_or_provide == "Filter"):

            printr("Currently Filtering!", True)
            for j in range(64):
                for i in range (64):
                    x = i + offset_x
                    y = j + offset_y

                    # Send rank 16 the filtered pixel
                    filtered_pixel_value = filter_pixel(x, y, offset_x, offset_y, offset_x + 64, offset_y + 64)

                    printr("Sending pixel value: " + str(filtered_pixel_value)+ " to 16 (Constructor)")

                    req = comm.isend([x, y, filtered_pixel_value], dest=16)
                    req.wait()
            
            # Tell our neighbor processes that were done filtering
            for neighbor in rank_neighbors[rank]:
                req = comm.isend(1, dest=neighbor)
                req.wait()

            # Tell controller were done filtering this sub image
            printr("Done filtering!", True)
            req = comm.isend(True, dest=17)
            req.wait()


        else:

            printr("Currently Providing!", True)
            neighbors_that_need_this_provider = 0
            neighbor_done_count = 0
            neighbors_in_order = []


            # Each provider process needs to know how many "done"
            # messages it needs to receive.
            for neighbor in rank_neighbors[rank]:
                if neighbor in order_arr[order_idx]:
                    neighbors_that_need_this_provider += 1
                    neighbors_in_order.append(neighbor)

            while neighbor_done_count != neighbors_that_need_this_provider:
                for neighbor in neighbors_in_order:

                    printr("Waiting on neighbor: " + str(neighbor))
                    if comm.Iprobe(source=neighbor):
                        req = comm.irecv(source=neighbor)
                        data = req.wait()

                        # We wait to receive data from filtering neighbor.
                        # If we recieve a done message, we just increment the done count,
                        # else we provide them with the requested pixel value
                        if data == 1:
                            neighbor_done_count += 1
                        else:
                            req = comm.isend(original_image[data[0], data[1]], dest=neighbor)
                            req.wait()
                        
            # Tell controller were done providing
            printr("Done Providing!", True)
            req = comm.isend(True, dest=17)
            req.wait()

        
        # Keep track of what layer we are in each individual process
        order_idx += 1
            
