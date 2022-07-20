# Description
Concurrent Programming assignment #4. This assignment was done in pairs: <br><br>
Jarrod Boone 201714680<br>
Yash Patel 201842812

## Implementation
We decided to just use point-to-point communication. In order to avoid deadlock, we filtered 4/16 subimages concurrently.
Every time we 

## Output
We included magick (https://imagemagick.org/script/index.php) which is used for editing digital images in this package. <br><br>
This allowed us to convert our output PGM to a PNG in order to see if we had similar outputs.
<br>
<br>
If your on Linux, to see the output, simply run ./magick output.pgm output.png to convert the program's output to a png.
<br>
<br>
Our output is shown below:
<br>
<br>

![output](output.png "Output")

## Requirements
This python script requires numpy and mpi4py

## Running the software
We were able to successfully generate the expected output using mpirun and mpiexec. The software requires 18 nodes (16 subimage nodes,
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

