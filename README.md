# Description
Concurrent Programming assignment #4. This assignment was done in pairs: <br><br>
Jarrod Boone 201714680<br>
Yash Patel 201842812

## Implementation
We used mainly point-to-point communication, where each subimage is givin it's own state during the filtering process. In order to avoid deadlock, we filtered 4/16 subimages concurrently and left the other 12 on standby to provide the neighboring pixels over MPI. After the 4 subimages are finished filtering, we then select another 4 subimages to filter, and another 12 subimages to provide. We repeat this 4 times in order to filter all 16 subimages.
<br><br>
We basically have 2 states for each subimage, "provide" and "filter". The `order_arr` shows the order in which each subimage will be filtered concurrently. <br>

As the subimages are being filtered using the gaussian kernel, they send their pixels to the 16th node for image reconstruction.
We also used a 17th node, which is responsible for assigning the state of each sub image.

## Output
We included the image edititng tool magick (https://imagemagick.org/script/index.php) in this package.
This allowed us to convert our output PGM to a PNG in order to see if we had similar outputs.
<br>
<br>
If your on Linux, to see the output, simply run `./magick output.pgm output.png` to convert the program's output to a png.
<br>
<br>
Our output is shown below:<br><br>
![output](output.png "Output")

## Requirements
This python script requires numpy and mpi4py.

## Running the software
We were able to successfully generate the expected output using mpirun and mpiexec. The program requires 18 nodes (16 subimage nodes,
a controller node, and a node responsible for reconstructing the image).
<br>
<br>
In order to run, we can use the following command:

`mpiexec -n 18 python main.py debug`
<br>
<br>
or
<br>
<br>
`mpirun -n 18 python main.py debug`

where debug is an optional parameter. If debug is passed in, the program will output the print statements of all 18 nodes
into corresponding text files (debug1.txt for node 1 for example).
